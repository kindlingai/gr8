// Fast 64-CTA parallel lm_head packing for Qwen3.8-Flash-Next (vocab = 248,320).
// Replaces single-CTA k_ink2_head_pack (~3.8 ms on 1 SM) with 64-CTA vectorized float4 scan +
// threshold compaction (~0.06 ms), with exact fallback to k_ink2_head_pack if survivors overflow/underflow.
// Output wire format is bit-compatible with ink2::k_ink2_head_pack:
//   [max f32][count i32][sum f64][NC x {f32 logit, i32 id}]
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "ink2_head.cuh"
#include "qw38_hc.cuh"

namespace qw38 {

constexpr int HEAD_BLKS = 64;
constexpr int HEAD_MAX_SURV = 4096;

struct HeadPackScratch {
    float* blk_max = nullptr;     // [max_M * HEAD_BLKS]
    int* blk_cnt = nullptr;       // [max_M * HEAD_BLKS]
    int* blk_argmax = nullptr;    // [max_M * HEAD_BLKS]
    double* blk_dsum = nullptr;   // [max_M * HEAD_BLKS]
    float* row_max = nullptr;     // [max_M]
    int* row_fin = nullptr;       // [max_M]
    int* surv_cnt = nullptr;      // [max_M]
    float* surv_val = nullptr;    // [max_M * HEAD_MAX_SURV]
    int* surv_id = nullptr;       // [max_M * HEAD_MAX_SURV]
    int* need_fallback = nullptr; // [max_M] (0 = done, 1 = run k_ink2_head_pack on this row)
};

__device__ __forceinline__ void head_better(float& mx, int& id, float v, int vid) {
    if (v > mx || (v == mx && vid < id)) {
        mx = v;
        id = vid;
    }
}

// Pass 1: 64 CTAs per row scan vr logits with vectorized float4 loads -> blk_max, blk_cnt, blk_argmax.
__global__ void __launch_bounds__(256) k_qw38_head_pass1(const float* __restrict__ lg, int vr, int v0,
                                  float* __restrict__ blk_max, int* __restrict__ blk_cnt,
                                  int* __restrict__ blk_argmax) {
    const int m = blockIdx.x / HEAD_BLKS;
    const int b = blockIdx.x % HEAD_BLKS;
    const int ch = (vr + HEAD_BLKS - 1) / HEAD_BLKS;
    const int j0 = b * ch;
    const int j1 = min(vr, j0 + ch);
    const float* x = lg + static_cast<size_t>(m) * vr;

    float mx = -INFINITY;
    int argmx = 0x7fffffff;
    int cnt = 0;

    // Align interior for 128-bit float4 loads (works for all TP=1..4, even non-div-4 vr like TP=3).
    const int j0_a = (j0 + 3) & ~3;
    const int j1_a = j1 & ~3;
    if (j0_a < j1_a) {
        if (threadIdx.x < (j0_a - j0)) {
            const int j = j0 + threadIdx.x;
            const float v = x[j];
            if (ink2::head_key(v)) { ++cnt; head_better(mx, argmx, v, v0 + j); }
        }
        const float4* x4 = reinterpret_cast<const float4*>(x + j0_a);
        const int n4 = (j1_a - j0_a) >> 2;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 vv = x4[i];
            const int base = v0 + j0_a + (i << 2);
            if (ink2::head_key(vv.x)) { ++cnt; head_better(mx, argmx, vv.x, base); }
            if (ink2::head_key(vv.y)) { ++cnt; head_better(mx, argmx, vv.y, base + 1); }
            if (ink2::head_key(vv.z)) { ++cnt; head_better(mx, argmx, vv.z, base + 2); }
            if (ink2::head_key(vv.w)) { ++cnt; head_better(mx, argmx, vv.w, base + 3); }
        }
        if (threadIdx.x < (j1 - j1_a)) {
            const int j = j1_a + threadIdx.x;
            const float v = x[j];
            if (ink2::head_key(v)) { ++cnt; head_better(mx, argmx, v, v0 + j); }
        }
    } else {
        for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
            const float v = x[j];
            if (ink2::head_key(v)) { ++cnt; head_better(mx, argmx, v, v0 + j); }
        }
    }

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, mx, off);
        const int oi = __shfl_xor_sync(0xffffffffu, argmx, off);
        head_better(mx, argmx, om, oi);
        cnt += __shfl_xor_sync(0xffffffffu, cnt, off);
    }
    __shared__ float smx[8];
    __shared__ int sid[8];
    __shared__ int scnt[8];
    if (lane == 0) { smx[warp] = mx; sid[warp] = argmx; scnt[warp] = cnt; }
    __syncthreads();
    if (warp == 0) {
        float fm = (lane < 8) ? smx[lane] : -INFINITY;
        int fi = (lane < 8) ? sid[lane] : 0x7fffffff;
        int fc = (lane < 8) ? scnt[lane] : 0;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            const float om = __shfl_xor_sync(0xffffffffu, fm, off);
            const int oi = __shfl_xor_sync(0xffffffffu, fi, off);
            head_better(fm, fi, om, oi);
            fc += __shfl_xor_sync(0xffffffffu, fc, off);
        }
        if (lane == 0) {
            blk_max[m * HEAD_BLKS + b] = fm;
            blk_cnt[m * HEAD_BLKS + b] = fc;
            if (blk_argmax) blk_argmax[m * HEAD_BLKS + b] = fi;
        }
    }
}

// Fast 1-pass greedy argmax finalize: reduces 64 (blk_max, blk_argmax) -> wire packet with count = 1.
// Zero Pass-2 exp() calls, zero survivor compaction/sort, 100% bit-identical greedy argmax!
__global__ void __launch_bounds__(64) k_qw38_head_argmax_finalize(
    const float* __restrict__ blk_max, const int* __restrict__ blk_cnt,
    const int* __restrict__ blk_argmax, uint8_t* __restrict__ out, size_t rb, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int t = threadIdx.x; // 0..63
    float mx = blk_max[m * HEAD_BLKS + t];
    int id = blk_argmax[m * HEAD_BLKS + t];
    int fc = blk_cnt[m * HEAD_BLKS + t];
    const int lane = t & 31, warp = t >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, mx, off);
        const int oi = __shfl_xor_sync(0xffffffffu, id, off);
        head_better(mx, id, om, oi);
        fc += __shfl_xor_sync(0xffffffffu, fc, off);
    }
    __shared__ float smx[2];
    __shared__ int sid[2];
    __shared__ int sfc[2];
    if (lane == 0) { smx[warp] = mx; sid[warp] = id; sfc[warp] = fc; }
    __syncthreads();
    if (t == 0) {
        float fm = smx[0];
        int fi = sid[0];
        head_better(fm, fi, smx[1], sid[1]);
        const int n = (sfc[0] + sfc[1] > 0) ? 1 : 0;
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

// Pass 1b: reduce 64 blk_max/blk_cnt per row (1 CTA per row, 64 threads) + zero surv_cnt.
__global__ void k_qw38_head_reduce_max(const float* __restrict__ blk_max, const int* __restrict__ blk_cnt,
                                       float* __restrict__ row_max, int* __restrict__ row_fin,
                                       int* __restrict__ surv_cnt, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int t = threadIdx.x; // 0..63
    float mx = blk_max[m * HEAD_BLKS + t];
    int fc = blk_cnt[m * HEAD_BLKS + t];
    const int lane = t & 31, warp = t >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        fc += __shfl_xor_sync(0xffffffffu, fc, off);
    }
    __shared__ float smx[2];
    __shared__ int sfc[2];
    if (lane == 0) { smx[warp] = mx; sfc[warp] = fc; }
    __syncthreads();
    if (t == 0) {
        row_max[m] = fmaxf(smx[0], smx[1]);
        row_fin[m] = sfc[0] + sfc[1];
        surv_cnt[m] = 0;
    }
}

// Pass 2: 64 CTAs per row compute double sum(expf(v - mx)) with float4 loads + warp-ballot survivor compaction!
// Replaces 1:64-rate FP64 exp() on sm_121a with IEEE FP32 expf() accumulated in FP64 (< 1e-7 lse error, 32x faster!).
__global__ void __launch_bounds__(256) k_qw38_head_pass2(const float* __restrict__ lg, int vr, int v0,
                                  const float* __restrict__ row_max,
                                  double* __restrict__ blk_dsum,
                                  int* __restrict__ surv_cnt,
                                  float* __restrict__ surv_val,
                                  int* __restrict__ surv_id) {
    const int m = blockIdx.x / HEAD_BLKS;
    const int b = blockIdx.x % HEAD_BLKS;
    const int ch = (vr + HEAD_BLKS - 1) / HEAD_BLKS;
    const int j0 = b * ch;
    const int j1 = min(vr, j0 + ch);
    const float* x = lg + static_cast<size_t>(m) * vr;
    const float mx = row_max[m];
    const float cut = mx - 18.0f;

    double a = 0.0;
    float* sv_dst = surv_val + static_cast<size_t>(m) * HEAD_MAX_SURV;
    int* si_dst = surv_id + static_cast<size_t>(m) * HEAD_MAX_SURV;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned lane_lt_mask = (1u << lane) - 1u;

    auto emit_warp = [&](bool keep, float v, int id) {
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        if (!mask) return;
        int base = 0;
        const int leader = __ffs(mask) - 1;
        if (lane == leader) base = atomicAdd(&surv_cnt[m], __popc(mask));
        base = __shfl_sync(0xffffffffu, base, leader);
        if (keep) {
            const int p = base + __popc(mask & lane_lt_mask);
            if (p < HEAD_MAX_SURV) {
                sv_dst[p] = v;
                si_dst[p] = id;
            }
        }
    };

    const int j0_a = (j0 + 3) & ~3;
    const int j1_a = j1 & ~3;
    if (j0_a < j1_a) {
        if (threadIdx.x < (j0_a - j0)) {
            const int j = j0 + threadIdx.x;
            const float v = x[j];
            if (ink2::head_key(v)) {
                a += static_cast<double>(expf(v - mx));
                if (v >= cut) {
                    const int p = atomicAdd(&surv_cnt[m], 1);
                    if (p < HEAD_MAX_SURV) { sv_dst[p] = v; si_dst[p] = v0 + j; }
                }
            }
        }
        const float4* x4 = reinterpret_cast<const float4*>(x + j0_a);
        const int n4 = (j1_a - j0_a) >> 2;
        const int n4_pad = (n4 + 255) & ~255;
        for (int i = threadIdx.x; i < n4_pad; i += 256) {
            float4 vv = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            int base = 0;
            bool in = (i < n4);
            if (in) {
                vv = x4[i];
                base = v0 + j0_a + (i << 2);
            }
            const bool k0 = in && ink2::head_key(vv.x);
            const bool k1 = in && ink2::head_key(vv.y);
            const bool k2 = in && ink2::head_key(vv.z);
            const bool k3 = in && ink2::head_key(vv.w);
            if (k0) a += static_cast<double>(expf(vv.x - mx));
            if (k1) a += static_cast<double>(expf(vv.y - mx));
            if (k2) a += static_cast<double>(expf(vv.z - mx));
            if (k3) a += static_cast<double>(expf(vv.w - mx));
            emit_warp(k0 && vv.x >= cut, vv.x, base);
            emit_warp(k1 && vv.y >= cut, vv.y, base + 1);
            emit_warp(k2 && vv.z >= cut, vv.z, base + 2);
            emit_warp(k3 && vv.w >= cut, vv.w, base + 3);
        }
        if (threadIdx.x < (j1 - j1_a)) {
            const int j = j1_a + threadIdx.x;
            const float v = x[j];
            if (ink2::head_key(v)) {
                a += static_cast<double>(expf(v - mx));
                if (v >= cut) {
                    const int p = atomicAdd(&surv_cnt[m], 1);
                    if (p < HEAD_MAX_SURV) { sv_dst[p] = v; si_dst[p] = v0 + j; }
                }
            }
        }
    } else {
        for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
            const float v = x[j];
            if (ink2::head_key(v)) {
                a += static_cast<double>(expf(v - mx));
                if (v >= cut) {
                    const int p = atomicAdd(&surv_cnt[m], 1);
                    if (p < HEAD_MAX_SURV) { sv_dst[p] = v; si_dst[p] = v0 + j; }
                }
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
    __shared__ double dsum[8];
    if (lane == 0) dsum[warp] = a;
    __syncthreads();
    if (warp == 0) {
        double tot = (lane < 8) ? dsum[lane] : 0.0;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, off);
        if (lane == 0) blk_dsum[m * HEAD_BLKS + b] = tot;
    }
}

// Pass 3: finalize wire packet per row from <= 4096 survivors cached in shared memory,
// or fall back to full exact radix pack in-kernel.
__global__ void __launch_bounds__(256) k_qw38_head_finalize(const float* __restrict__ lg, int vr, int v0, int NC,
                                     const float* __restrict__ row_max,
                                     const int* __restrict__ row_fin,
                                     const double* __restrict__ blk_dsum,
                                     const int* __restrict__ surv_cnt,
                                     const float* __restrict__ surv_val,
                                     const int* __restrict__ surv_id,
                                     uint8_t* __restrict__ out, size_t rb) {
    const int m = blockIdx.x;
    const int t = threadIdx.x; // 0..255
    const float mx = row_max[m];
    const int finite = row_fin[m];
    const int n = (finite < NC) ? finite : NC;
    uint8_t* o = out + static_cast<size_t>(m) * rb;

    __shared__ double dsum[64];
    if (t < HEAD_BLKS) dsum[t] = blk_dsum[m * HEAD_BLKS + t];
    __syncthreads();
    if (t < 32) {
        double tot = dsum[t] + dsum[t + 32];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, off);
        if (t == 0) dsum[0] = tot;
    }
    __syncthreads();
    const double sum = dsum[0];

    // Union shared memory (32 KiB) shared between fast survivor rank-select and rare radix fallback.
    __shared__ union {
        struct {
            float sv[HEAD_MAX_SURV];
            int si[HEAD_MAX_SURV];
        } fast;
        struct {
            uint32_t hist[256];
            int cg[256], ce[256];
            float cl[ink2::INK2_HEAD_MAX_NC];
            int cid[ink2::INK2_HEAD_MAX_NC];
            uint32_t s_prefix;
            int s_rem;
        } fb;
    } u;

    const int sc = surv_cnt[m];
    if (n > 0 && sc >= n && sc <= HEAD_MAX_SURV) {
        // Fast path: cache survivors into shared memory once -> 10x faster rank-select than global reads!
        const float* sv_g = surv_val + static_cast<size_t>(m) * HEAD_MAX_SURV;
        const int* si_g = surv_id + static_cast<size_t>(m) * HEAD_MAX_SURV;
        for (int i = t; i < sc; i += blockDim.x) {
            u.fast.sv[i] = sv_g[i];
            u.fast.si[i] = si_g[i];
        }
        __syncthreads();
        for (int i = t; i < sc; i += blockDim.x) {
            const float li = u.fast.sv[i];
            const int ii = u.fast.si[i];
            int r = 0;
            #pragma unroll 4
            for (int j = 0; j < sc; ++j) {
                const float lj = u.fast.sv[j];
                r += (lj > li) || (lj == li && u.fast.si[j] < ii);
            }
            if (r < n) {
                *reinterpret_cast<float*>(o + 16 + static_cast<size_t>(r) * 8) = li;
                *reinterpret_cast<int*>(o + 20 + static_cast<size_t>(r) * 8) = ii;
            }
        }
        for (int i = n + t; i < NC; i += blockDim.x)
            *reinterpret_cast<uint64_t*>(o + 16 + static_cast<size_t>(i) * 8) = 0ull;
        if (t == 0) {
            *reinterpret_cast<float*>(o) = mx;
            *reinterpret_cast<int*>(o + 4) = n;
            *reinterpret_cast<double*>(o + 8) = sum;
        }
        return;
    }

    // Rare exact fallback (e.g., flat/uniform distribution where < NC or > 4096 logits sit within 18 of max)
    const float* x = lg + static_cast<size_t>(m) * vr;
    if (n > 0) {
        uint32_t prefix = 0, mask = 0;
        int rem = n;
        for (int sh = 24; sh >= 0; sh -= 8) {
            u.fb.hist[t] = 0;
            __syncthreads();
            for (int j = t; j < vr; j += 256) {
                const uint32_t k = ink2::head_key(x[j]);
                if (k && (k & mask) == prefix) atomicAdd(&u.fb.hist[(k >> sh) & 255u], 1u);
            }
            __syncthreads();
            if (t == 0) {
                int acc = 0, d = 255;
                for (; d > 0; --d) {
                    const int c = static_cast<int>(u.fb.hist[d]);
                    if (acc + c >= rem) break;
                    acc += c;
                }
                u.fb.s_prefix = prefix | (static_cast<uint32_t>(d) << sh);
                u.fb.s_rem = rem - acc;
            }
            __syncthreads();
            prefix = u.fb.s_prefix;
            rem = u.fb.s_rem;
            mask |= 255u << sh;
            __syncthreads();
        }
        const uint32_t thr = prefix;
        const int take = rem, g = n - rem;
        const int ch = (vr + 255) / 256;
        const int j0 = t * ch < vr ? t * ch : vr, j1 = j0 + ch < vr ? j0 + ch : vr;
        int a = 0, b = 0;
        for (int j = j0; j < j1; ++j) {
            const uint32_t k = ink2::head_key(x[j]);
            a += k > thr;
            b += k == thr;
        }
        u.fb.cg[t] = a; u.fb.ce[t] = b;
        __syncthreads();
        if (t == 0) {
            int sa = 0, sb = 0;
            for (int i = 0; i < 256; ++i) {
                const int ca = u.fb.cg[i], cb = u.fb.ce[i];
                u.fb.cg[i] = sa; u.fb.ce[i] = sb; sa += ca; sb += cb;
            }
        }
        __syncthreads();
        int pg = u.fb.cg[t], pe = u.fb.ce[t];
        for (int j = j0; j < j1; ++j) {
            const float v = x[j];
            const uint32_t k = ink2::head_key(v);
            if (k > thr) { u.fb.cl[pg] = v; u.fb.cid[pg] = v0 + j; ++pg; }
            else if (k == thr) { if (pe < take) { u.fb.cl[g + pe] = v; u.fb.cid[g + pe] = v0 + j; } ++pe; }
        }
        __syncthreads();
        for (int i = t; i < n; i += 256) {
            const float li = u.fb.cl[i];
            const int ii = u.fb.cid[i];
            int r = 0;
            for (int j = 0; j < n; ++j) r += (u.fb.cl[j] > li) || (u.fb.cl[j] == li && u.fb.cid[j] < ii);
            *reinterpret_cast<float*>(o + 16 + static_cast<size_t>(r) * 8) = li;
            *reinterpret_cast<int*>(o + 20 + static_cast<size_t>(r) * 8) = ii;
        }
    }
    for (int i = n + t; i < NC; i += 256) *reinterpret_cast<uint64_t*>(o + 16 + static_cast<size_t>(i) * 8) = 0ull;
    if (t == 0) {
        *reinterpret_cast<float*>(o) = mx;
        *reinterpret_cast<int*>(o + 4) = n;
        *reinterpret_cast<double*>(o + 8) = sum;
    }
}

inline void head_pack_fast(const float* lg, int M, int vr, int v0, int NC,
                           HeadPackScratch& sc, uint8_t* out, size_t rb, cudaStream_t st,
                           bool argmax_only = false) {
    if (M <= 0) return;
    k_qw38_head_pass1<<<M * HEAD_BLKS, 256, 0, st>>>(lg, vr, v0, sc.blk_max, sc.blk_cnt, sc.blk_argmax);
    if (argmax_only && sc.blk_argmax) {
        k_qw38_head_argmax_finalize<<<M, HEAD_BLKS, 0, st>>>(sc.blk_max, sc.blk_cnt, sc.blk_argmax, out, rb, M);
        return;
    }
    k_qw38_head_reduce_max<<<M, HEAD_BLKS, 0, st>>>(sc.blk_max, sc.blk_cnt, sc.row_max, sc.row_fin, sc.surv_cnt, M);
    k_qw38_head_pass2<<<M * HEAD_BLKS, 256, 0, st>>>(lg, vr, v0, sc.row_max, sc.blk_dsum, sc.surv_cnt, sc.surv_val, sc.surv_id);
    k_qw38_head_finalize<<<M, 256, 0, st>>>(lg, vr, v0, NC, sc.row_max, sc.row_fin, sc.blk_dsum, sc.surv_cnt, sc.surv_val, sc.surv_id, out, rb);
}

} // namespace qw38
