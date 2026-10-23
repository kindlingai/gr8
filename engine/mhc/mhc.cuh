// Flash mHC kernels. No effect on the legacy GLM build unless included.
#pragma once
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "../model_geom.h"
#include "math.h"
namespace flash_mhc {
constexpr int THREADS = 256;
struct Weights {
    const __nv_bfloat16* fn = nullptr; // [24,4*H]; BF16
    const float *base = nullptr, *scale = nullptr; // [24], [3]; FP32
    const __nv_bfloat16* norm = nullptr;                        // [H]; BF16
};
struct Buffers {
    float* streams; // [max_rows,4,H], FP32; updated in place, never narrowed
    float* inv;     // [max_rows], flat RMS inverse, refreshed by embed or consume
    float* mix;     // [max_rows,24]
    float* pre;    // [max_rows,4]
    float* post;   // [max_rows,4]
    float* comb;   // [max_rows,4,4], source then destination
    unsigned* done = nullptr; // [max_rows], zeroed retirement counter for fused prepare
    int bf16 = 0;             // test (G6_MHC_BF16): this launch rounds every stream write to BF16 (= BF16 storage, FP32 math)
};
// G6_MHC_BF16: quality test of BF16 residual streams. The buffer stays FP32; each written value is rounded to BF16, so
// every later read, sum of squares and register copy sees exactly what a BF16 buffer would hold. 1 = the writes after
// both blocks, 2 = only the write after the MLP block (attention-side write stays FP32). Default 0 = unchanged.
inline int bf16_env() {   // bit 0 = attention-side write, bit 1 = MLP-side write
    static const int mask = [] {
        const int v = getenv("G6_MHC_BF16") ? atoi(getenv("G6_MHC_BF16")) : 0;
        return v == 1 ? 3 : v == 2 ? 2 : 0;
    }();
    return mask;
}
// The buffers for the stream write after block mlp (0 = attention, 1 = MLP).
inline Buffers side(Buffers b, int mlp) {
    b.bf16 = (bf16_env() >> mlp) & 1;
    return b;
}
__device__ __forceinline__ float narrow(const Buffers& b, float v) {
    return b.bf16 ? __bfloat162float(__float2bfloat16_rn(v)) : v;
}
__device__ inline float sum_block(float v) {
    __shared__ float sums[8];
    for (int d=16; d; d/=2) v += __shfl_down_sync(0xffffffffu, v, d);
    if (!(threadIdx.x & 31)) sums[threadIdx.x/32] = v;
    __syncthreads();
    v = threadIdx.x < 8 ? sums[threadIdx.x] : 0.f;
    if (threadIdx.x < 32) {
        for (int d=16; d; d/=2) v += __shfl_down_sync(0xffffffffu, v, d);
        if (!threadIdx.x) sums[0] = v;
    }
    __syncthreads();
    return sums[0];
}
__device__ inline void write_frag_pair(int m, int M, int H, int h, float a, float b, unsigned* f16, unsigned* bf16) {
    const size_t dst = (size_t)(m/8)*(H/16)*64 + ((h/16)*32 + (m%8)*4 + (h%8)/2)*2 + (h/8)%2;
    if (f16) f16[dst] = (unsigned)__half_as_ushort(__float2half_rn(a)) |
                        ((unsigned)__half_as_ushort(__float2half_rn(b))<<16);
    if (bf16) bf16[dst] = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(a)) |
                          ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(b))<<16);
    if (m + 1 == M) {
        const int padded = (M + 7) / 8 * 8;
        for (int mp = M; mp < padded; ++mp) {
            const size_t pdst = dst + (size_t)(mp - m) * 8;
            if (f16) f16[pdst] = 0u;
            if (bf16) bf16[pdst] = 0u;
        }
    }
}
// Warp-parallel 4x4 Sinkhorn coefficients: lanes 0..15 hold comb[4*i+j] and reduce rows/cols
// in the exact left-to-right order of flash_mhc::coefficients.
__device__ inline void coefficients_warp(const volatile float* mix, const float* base, const float* scale,
                                         float* pre, float* post, float* comb, float* sh_pre = nullptr) {
    const int k = threadIdx.x;
    if (k < 16) {
        const int i = k >> 2, j = k & 3;
        if (k < 4) {
            const float p = sigmoid(mix[k] * scale[0] + base[k]) + HC_EPS;
            if (sh_pre) sh_pre[k] = p;
            pre[k] = p;
            post[k] = 2.f * sigmoid(mix[4 + k] * scale[1] + base[4 + k]);
        }
        float c = mix[8 + k] * scale[2] + base[8 + k];
        float mx = -INFINITY;
        for (int jj = 0; jj < 4; ++jj) mx = fmaxf(mx, __shfl_sync(0xffffu, c, 4*i + jj));
        c = expf(c - mx);
        float sum = 0.f;
        for (int jj = 0; jj < 4; ++jj) sum += __shfl_sync(0xffffu, c, 4*i + jj);
        c = c / sum + HC_EPS;
        for (int iter = 0; iter < ITERS; ++iter) {
            if (iter) {
                float rsum = 0.f;
                for (int jj = 0; jj < 4; ++jj) rsum += __shfl_sync(0xffffu, c, 4*i + jj);
                c /= rsum + HC_EPS;
            }
            float csum = 0.f;
            for (int ii = 0; ii < 4; ++ii) csum += __shfl_sync(0xffffu, c, 4*ii + j);
            c /= csum + HC_EPS;
        }
        comb[k] = c;
    }
}
__device__ inline void collapse_norm_row(Buffers b, Weights w, int m, int M, int H, float eps,
                                         const float* pre4, float* row, __half* out,
                                         float* raw, float* normalized, unsigned* f16, unsigned* bf16) {
    const float* __restrict__ s = b.streams + (size_t)m * 4 * H;
    float sq = 0.f;
    for (int h = threadIdx.x; h < H; h += THREADS) {
        float v = 0.f;
        for (int j = 0; j < 4; ++j) v += pre4[j] * s[j*H + h];
        row[h] = v; sq += v * v;
        if (raw) raw[(size_t)m*H + h] = v;
    }
    const float inv = rsqrtf(sum_block(sq) / H + eps);
    for (int h = threadIdx.x; h < H; h += THREADS) {
        const float v = row[h] * inv * __bfloat162float(w.norm[h]);
        if (out) out[(size_t)m*H + h] = __float2half_rn(v);
        if (normalized) normalized[(size_t)m*H + h] = v;
        if (f16 || bf16) {
            const float b_val = __shfl_xor_sync(__activemask(), v, 1);
            if (!(threadIdx.x & 1)) write_frag_pair(m, M, H, h, v, b_val, f16, bf16);
        }
    }
}
// One CTA per row. Embedding copy and initial flat RMS in one pass.
__global__ void k_mhc_embed(Buffers b, const int* tok, const __nv_bfloat16* embed, int H, float eps) {
    const int m=blockIdx.x;
    float sq=0.f;
    for (int h=threadIdx.x; h<H; h+=THREADS) {
        const float v=__bfloat162float(embed[(size_t)tok[m]*H+h]);
        for (int j=0; j<4; ++j) b.streams[((size_t)m*4+j)*H+h]=v;
        sq += v*v;
    }
    const float inv=rsqrtf(sum_block(sq)/H+eps);
    if (!threadIdx.x) b.inv[m]=inv;
}
// Standalone entry for golden vectors and residual restores.
__global__ void k_mhc_flat_inv(Buffers b, int H, float eps) {
    float sq=0.f;
    for (int k=threadIdx.x; k<4*H; k+=THREADS) {
        const float v=b.streams[(size_t)blockIdx.x*4*H+k]; sq+=v*v;
    }
    const float inv=rsqrtf(sum_block(sq)/(4*H)+eps);
    if (!threadIdx.x) b.inv[blockIdx.x]=inv;
}
// grid(24,M): FP32 GEMV; normalize each input BEFORE multiplying its weight.
__global__ void k_mhc_mix(Buffers b, Weights w, int H) {
    const int o=blockIdx.x, m=blockIdx.y;
    const float inv=b.inv[m];
    const float* __restrict__ s=b.streams+(size_t)m*4*H;
    const __nv_bfloat16* __restrict__ fn=w.fn+(size_t)o*4*H;
    float sum=0.f;
    for (int k=threadIdx.x; k<4*H; k+=THREADS)
        sum += (s[k]*inv) * __bfloat162float(fn[k]);
    const float v=sum_block(sum);
    if (!threadIdx.x) b.mix[m*24+o]=v;
}
__global__ void k_mhc_coefficients(Buffers b, Weights w) {
    const int m=blockIdx.x;
    coefficients_warp(b.mix+24*m,w.base,w.scale,b.pre+4*m,b.post+4*m,b.comb+16*m);
}
// Collapse and block RMSNorm, with optional FP32 golden outputs before narrowing and optional fragment writes.
__global__ void k_mhc_collapse_norm(Buffers b, Weights w, int H, float eps,
                                   __half* out, float* raw=nullptr, float* normalized=nullptr,
                                   unsigned* f16=nullptr, unsigned* bf16=nullptr, bool do_coeff=false) {
    extern __shared__ float row[];
    __shared__ float sh_pre[4];
    const int m=blockIdx.x, M=gridDim.x;
    if (do_coeff) {
        coefficients_warp(b.mix+24*m,w.base,w.scale,b.pre+4*m,b.post+4*m,b.comb+16*m,sh_pre);
        __syncthreads();
    } else {
        if (threadIdx.x < 4) sh_pre[threadIdx.x] = b.pre[m*4+threadIdx.x];
        __syncthreads();
    }
    collapse_norm_row(b,w,m,M,H,eps,sh_pre,row,out,raw,normalized,f16,bf16);
}
// F// Fused grid(24,M): all 24 CTAs compute mix[m,o]; the last CTA for row m runs
// coefficients + collapse + RMSNorm + fragment tables in a single launch.
__device__ inline void write_frag_pair_nozero(int m, int H, int h, float a, float b, unsigned* f16, unsigned* bf16) {
    const size_t dst = (size_t)(m/8)*(H/16)*64 + ((h/16)*32 + (m%8)*4 + (h%8)/2)*2 + (h/8)%2;
    if (f16) f16[dst] = (unsigned)__half_as_ushort(__float2half_rn(a)) |
                        ((unsigned)__half_as_ushort(__float2half_rn(b))<<16);
    if (bf16) bf16[dst] = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(a)) |
                          ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(b))<<16);
}
// row body shared by k_mhc_prepare_fused and its owner-range (Flash SP, G6_PF_RS=2) entry point: blockIdx.y indexes
// rows of the LAUNCHED range, m is the GLOBAL row (first + blockIdx.y); M_lane is the FULL lane row count (gridDim.y
// on the OFF entry, M on the SP entry), used only by the fragment zero pad and the non-4096 collapse path.
__device__ __forceinline__ void mhc_prepare_fused_body(Buffers b, Weights w, int first, int M_lane, int H, float eps,
                                                       __half* out, float* normalized, unsigned* f16, unsigned* bf16) {
    __shared__ float sh_pre[4];
    __shared__ int sh_last;
    const int o=blockIdx.x, m=first+blockIdx.y, M=M_lane;
    const float inv=b.inv[m];
    const float* __restrict__ s=b.streams+(size_t)m*4*H;
    const __nv_bfloat16* __restrict__ fn=w.fn+(size_t)o*4*H;
    float sum=0.f;
    if (H == 4096) {
        #pragma unroll 16
        for (int idx=0; idx<64; ++idx) {
            const int k=threadIdx.x + idx*THREADS;
            sum += (__ldg(s+k)*inv) * __bfloat162float(__ldg(fn+k));
        }
    } else {
        #pragma unroll 8
        for (int k=threadIdx.x; k<4*H; k+=THREADS)
            sum += (s[k]*inv) * __bfloat162float(fn[k]);
    }
    const float v=sum_block(sum);
    if (!threadIdx.x) {
        b.mix[m*24+o]=v;
        __threadfence();
        sh_last = (atomicInc(b.done+m, MIX-1) == MIX-1);
    }
    if (H == 4096 && m + 1 == M && (f16 || bf16)) {
        const int padded = (M + 7) / 8 * 8;
        const int mp = M + o;
        if (mp < padded) {
            const int ks = threadIdx.x;
            const size_t base = (size_t)(mp / 8) * 256 * 64 + (ks * 32 + (mp % 8) * 4) * 2;
            if (f16) {
                uint4* d = reinterpret_cast<uint4*>(f16 + base);
                d[0] = make_uint4(0u, 0u, 0u, 0u);
                d[1] = make_uint4(0u, 0u, 0u, 0u);
            }
            if (bf16) {
                uint4* d = reinterpret_cast<uint4*>(bf16 + base);
                d[0] = make_uint4(0u, 0u, 0u, 0u);
                d[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }
    __syncthreads();
    if (!sh_last) return;
    coefficients_warp(b.mix+24*m,w.base,w.scale,b.pre+4*m,b.post+4*m,b.comb+16*m,sh_pre);
    __syncthreads();
    if (H == 4096) {
        const float p0 = sh_pre[0], p1 = sh_pre[1], p2 = sh_pre[2], p3 = sh_pre[3];
        float sq = 0.f;
        float rv[16];
        #pragma unroll
        for (int idx = 0; idx < 16; ++idx) {
            const int h = threadIdx.x + idx * THREADS;
            float val = 0.f;
            val += p0 * __ldg(s + h);
            val += p1 * __ldg(s + 4096 + h);
            val += p2 * __ldg(s + 8192 + h);
            val += p3 * __ldg(s + 12288 + h);
            rv[idx] = val;
            sq += val * val;
        }
        const float rinv = rsqrtf(sum_block(sq) / 4096 + eps);
        #pragma unroll
        for (int idx = 0; idx < 16; ++idx) {
            const int h = threadIdx.x + idx * THREADS;
            const float val = rv[idx] * rinv * __bfloat162float(__ldg(w.norm + h));
            if (out) out[(size_t)m * 4096 + h] = __float2half_rn(val);
            if (normalized) normalized[(size_t)m * 4096 + h] = val;
            if (f16 || bf16) {
                const float b_val = __shfl_xor_sync(0xffffffffu, val, 1);
                if (!(threadIdx.x & 1)) write_frag_pair_nozero(m, 4096, h, val, b_val, f16, bf16);
            }
        }
    } else {
        extern __shared__ float row[];
        collapse_norm_row(b,w,m,M,H,eps,sh_pre,row,out,nullptr,normalized,f16,bf16);
    }
}
__global__ void __launch_bounds__(THREADS) k_mhc_prepare_fused(Buffers b, Weights w, int H, float eps,
                                    __half* out, float* normalized=nullptr,
                                    unsigned* f16=nullptr, unsigned* bf16=nullptr) {
    mhc_prepare_fused_body(b, w, 0, gridDim.y, H, eps, out, normalized, f16, bf16);
}
// owner-range entry (Flash SP, G6_PF_RS=2): grid dim3(MIX, rows) over GLOBAL rows [first, mend). Fragment tables
// are not built on the SP path (it rejects G6_PF_XFRAG), so f16/bf16 are always null; M_lane stays the full-lane M.
__global__ void __launch_bounds__(THREADS) k_mhc_prepare_fused_sp(Buffers b, Weights w, int first, int M_lane, int H, float eps,
                                    __half* out, float* normalized) {
    mhc_prepare_fused_body(b, w, first, M_lane, H, eps, out, normalized, nullptr, nullptr);
}
// Getter reads a reduced block output. Its rank-order reduction can read fabric
// slots directly, so no intermediate reduced [M,H] allocation is necessary.
struct Fp32Block {
    const float* p;
    __device__ unsigned long long init() { return 0ULL; }
    __device__ void wait(unsigned long long) const {}
    __device__ unsigned seq32(unsigned long long) const { return 1u; }
    __device__ float operator()(size_t k) const { return p[k]; }
};
struct Bf16Partials {
    const __nv_bfloat16* const* peers; // device pointer table [nr], each [M,H]
    int nr;
    __device__ unsigned long long init() { return 0ULL; }
    __device__ void wait(unsigned long long) const {}
    __device__ unsigned seq32(unsigned long long) const { return 1u; }
    __device__ float operator()(size_t k) const {
        float v=0.f;
        for (int p=0; p<nr; ++p) v+=__bfloat162float(peers[p][k]);
        return v;
    }
};
// row body shared by k_mhc_out_inv and its owner-range (Flash SP) entry point: m is the GLOBAL lane row, first the
// first row of the launched range (block index k is owner-local: (m - first) * H + h). OFF passes first = 0.
template<class Block>
__device__ __forceinline__ void mhc_out_inv_body(Buffers b, Block& block, int m, int first, int H, float eps) {
    const unsigned long long step = block.init();
    float post[4], comb[16];
    #pragma unroll
    for (int i=0; i<4; ++i) post[i]=b.post[m*4+i];
    #pragma unroll
    for (int k=0; k<16; ++k) comb[k]=b.comb[m*16+k];
    float* __restrict__ streams = b.streams + (size_t)m*4*H;
    float sq=0.f;
    if (H == 4096) {
        float mix_all[16][4];
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            float old[4];
            #pragma unroll
            for (int j=0; j<4; ++j) old[j]=__ldg(streams+j*4096+h);
            #pragma unroll
            for (int i=0; i<4; ++i) {
                float mix=0.f;
                #pragma unroll
                for (int j=0; j<4; ++j) mix+=comb[j*4+i]*old[j];
                mix_all[idx][i]=mix;
            }
        }
        block.wait(step);
        float y_all[16];
        #pragma unroll
        for (int idx=0; idx<16; ++idx)
            y_all[idx] = block((size_t)(m-first)*4096 + threadIdx.x + idx*THREADS);
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            const float y = y_all[idx];
            #pragma unroll
            for (int i=0; i<4; ++i) {
                const float v=narrow(b, post[i]*y+mix_all[idx][i]);
                streams[i*4096+h]=v; sq+=v*v;
            }
        }
    } else {
        block.wait(step);
        for (int h=threadIdx.x; h<H; h+=THREADS) {
            float old[4];
            #pragma unroll
            for (int j=0; j<4; ++j) old[j]=streams[j*H+h];
            const float y=block((size_t)(m-first)*H+h);
            #pragma unroll
            for (int i=0; i<4; ++i) {
                float mix=0.f;
                #pragma unroll
                for (int j=0; j<4; ++j) mix+=comb[j*4+i]*old[j];
                const float v=narrow(b, post[i]*y+mix);
                streams[i*H+h]=v; sq+=v*v;
            }
        }
    }
    const float inv=rsqrtf(sum_block(sq)/(4*H)+eps);
    if (!threadIdx.x) b.inv[m]=inv;
}
template<class Block> __global__ void __launch_bounds__(THREADS) k_mhc_out_inv(Buffers b, Block block, int H, float eps) {
    mhc_out_inv_body(b, block, blockIdx.x, 0, H, eps);
}
// owner-range entry (Flash SP, G6_PF_RS=2): one CTA per GLOBAL row [first, mend); grid = mend - first (>= 1).
// All stream/scratch/output addressing stays by the global row m; the block reads owner-local elements.
template<class Block> __global__ void __launch_bounds__(THREADS) k_mhc_out_inv_sp(Buffers b, Block block, int first, int H, float eps) {
    mhc_out_inv_body(b, block, first + blockIdx.x, first, H, eps);
}
// Single-CTA-per-row fused wait_peers + out_inv + mix + coefficients + collapse_norm + fragments for H==4096.
// Keeps updated streams[m] in registers (mix_all[16][4]) across all 24 mix dot products and collapse_norm,
// eliminating all 25 global memory re-reads of b.streams (3.35 GB per collective at M=2048). 0-ULP bit-identical.
#ifndef MHC_RR_UNROLL
#define MHC_RR_UNROLL 8   // k-loop unroll of k_mhc_out_inv_prepare_rr (bench knob)
#endif
#ifndef MHC_OIP_MINB
#define MHC_OIP_MINB 1   // CTAs per SM the register budget of k_mhc_out_inv_prepare targets (bench knob)
#endif
template<class Block>
__global__ void __launch_bounds__(THREADS, MHC_OIP_MINB) k_mhc_out_inv_prepare(
    Buffers b, Block block, Weights w, int H, float eps,
    __half* out, float* normalized=nullptr, unsigned* f16=nullptr, unsigned* bf16=nullptr) {
    __shared__ float sh_sums[8][8];
    __shared__ float sh_mix[24];
    __shared__ float sh_pre[4];
    const int m=blockIdx.x, M=gridDim.x;
    const unsigned long long step = block.init();
    float post[4], comb[16];
    #pragma unroll
    for (int i=0; i<4; ++i) post[i]=b.post[m*4+i];
    #pragma unroll
    for (int k=0; k<16; ++k) comb[k]=b.comb[m*16+k];
    float* __restrict__ streams = b.streams + (size_t)m*4*H;
    float mix_all[16][4];
    #pragma unroll
    for (int idx=0; idx<16; ++idx) {
        const int h = threadIdx.x + idx*THREADS;
        float old[4];
        #pragma unroll
        for (int j=0; j<4; ++j) old[j]=__ldg(streams+j*4096+h);
        #pragma unroll
        for (int i=0; i<4; ++i) {
            float mix=0.f;
            #pragma unroll
            for (int j=0; j<4; ++j) mix+=comb[j*4+i]*old[j];
            mix_all[idx][i]=mix;
        }
    }
    block.wait(step);
    float sq=0.f;
    #pragma unroll
    for (int idx=0; idx<16; ++idx) {
        const int h = threadIdx.x + idx*THREADS;
        const float y = block((size_t)m*4096 + h);
        #pragma unroll
        for (int i=0; i<4; ++i) {
            const float v=narrow(b, post[i]*y+mix_all[idx][i]);
            streams[i*4096+h]=v;
            mix_all[idx][i]=v;
            sq+=v*v;
        }
    }
    const float inv=rsqrtf(sum_block(sq)/(4*4096)+eps);
    if (!threadIdx.x) b.inv[m]=inv;
    #pragma unroll 1
    for (int pass=0; pass<3; ++pass) {
        float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const __nv_bfloat16* __restrict__ fn_p = w.fn + (size_t)(pass*8)*16384 + threadIdx.x;
        #pragma unroll
        for (int i=0; i<4; ++i) {
            #pragma unroll
            for (int idx=0; idx<16; ++idx) {
                const int off = i*4096 + idx*THREADS;
                const float si = mix_all[idx][i] * inv;
                #pragma unroll
                for (int c=0; c<8; ++c)
                    sum[c] += si * __bfloat162float(__ldg(fn_p + c*16384 + off));
            }
        }
        #pragma unroll
        for (int c=0; c<8; ++c) {
            float v = sum[c];
            for (int d=16; d; d/=2) v += __shfl_down_sync(0xffffffffu, v, d);
            if (!(threadIdx.x & 31)) sh_sums[c][threadIdx.x >> 5] = v;
        }
        __syncthreads();
        const int c = threadIdx.x >> 5, lane = threadIdx.x & 31;
        float v = lane < 8 ? sh_sums[c][lane] : 0.f;
        for (int d=16; d; d/=2) v += __shfl_down_sync(0xffffffffu, v, d);
        if (!lane) {
            sh_mix[pass*8+c] = v;
            b.mix[m*24+pass*8+c] = v;
        }
        __syncthreads();
    }
    coefficients_warp(sh_mix,w.base,w.scale,b.pre+4*m,b.post+4*m,b.comb+16*m,sh_pre);
    __syncthreads();
    const float p0 = sh_pre[0], p1 = sh_pre[1], p2 = sh_pre[2], p3 = sh_pre[3];
    float csq = 0.f;
    float rv[16];
    #pragma unroll
    for (int idx=0; idx<16; ++idx) {
        float val = 0.f;
        val += p0 * mix_all[idx][0];
        val += p1 * mix_all[idx][1];
        val += p2 * mix_all[idx][2];
        val += p3 * mix_all[idx][3];
        rv[idx] = val;
        csq += val * val;
    }
    const float rinv = rsqrtf(sum_block(csq) / 4096 + eps);
    #pragma unroll
    for (int idx=0; idx<16; ++idx) {
        const int h = threadIdx.x + idx*THREADS;
        const float val = rv[idx] * rinv * __bfloat162float(__ldg(w.norm + h));
        if (out) out[(size_t)m * 4096 + h] = __float2half_rn(val);
        if (normalized) normalized[(size_t)m * 4096 + h] = val;
        if (f16 || bf16) {
            const float b_val = __shfl_xor_sync(0xffffffffu, val, 1);
            if (!(threadIdx.x & 1)) write_frag_pair(m, M, 4096, h, val, b_val, f16, bf16);
        }
    }
}
// fn [24][4*4096] BF16 -> packed [3 pass][4 i][16 idx][256 thread] x uint4 (the 8 c of one pass): the element thread t
// reads at (pass, c, i, idx) in k_mhc_out_inv_prepare, 8 per 16-B load. 49152 uint4 = 768 KiB per Weights.
__global__ void k_mhc_pack_fn(const __nv_bfloat16* __restrict__ fn, uint4* __restrict__ fnp) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= 3 * 4 * 16 * THREADS) return;
    const int t = q % THREADS, idx = (q / THREADS) % 16, i = (q / (THREADS * 16)) % 4, pass = q / (THREADS * 64);
    unsigned short v[8];
    #pragma unroll
    for (int c=0; c<8; ++c)
        v[c] = __bfloat16_as_ushort(fn[(size_t)(pass*8 + c)*16384 + i*4096 + idx*THREADS + t]);
    uint4 o;
    o.x = v[0] | ((unsigned)v[1] << 16); o.y = v[2] | ((unsigned)v[3] << 16);
    o.z = v[4] | ((unsigned)v[5] << 16); o.w = v[6] | ((unsigned)v[7] << 16);
    fnp[q] = o;
}
// k_mhc_out_inv_prepare with the packed fn (k_mhc_pack_fn) and R rows per CTA (rows m0 .. m0 + R - 1 of the
// LAUNCHED range; grid ceil(rows / R)): each 16-B fn load serves 8 c and R rows. Same arithmetic in the same order
// per row -> bit-identical. The row body is shared with the owner-range (Flash SP) entry point below: m0 = first +
// blockIdx.x * R over GLOBAL rows [first, mend) of a full lane, the block index is owner-local ((m - first) * 4096),
// and every stream/scratch/output address uses the global row m. OFF passes first = 0, mend = M.
template<class Block, int R>
__device__ __forceinline__ void mhc_out_inv_prepare_pk_body(
    Buffers b, Block& block, Weights w, const uint4* __restrict__ fnp, int first, int mend, float eps,
    __half* out, float* normalized) {
    __shared__ float sh_sums[R][8][8];
    __shared__ float sh_mix[R][24];
    __shared__ float sh_pre[R][4];
    const int m0 = first + blockIdx.x * R;
    const int rr = mend - m0 < R ? mend - m0 : R;   // owner-end guard, uniform over the CTA
    const unsigned long long step = block.init();
    float mix_all[R][16][4];
    #pragma unroll
    for (int r=0; r<R; ++r) {
        if (r >= rr) break;
        const int m = m0 + r;
        float comb[16];
        #pragma unroll
        for (int k=0; k<16; ++k) comb[k]=b.comb[m*16+k];
        const float* __restrict__ streams = b.streams + (size_t)m*4*4096;
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            float old[4];
            #pragma unroll
            for (int j=0; j<4; ++j) old[j]=__ldg(streams+j*4096+h);
            #pragma unroll
            for (int i=0; i<4; ++i) {
                float mix=0.f;
                #pragma unroll
                for (int j=0; j<4; ++j) mix+=comb[j*4+i]*old[j];
                mix_all[r][idx][i]=mix;
            }
        }
    }
    block.wait(step);
    float inv[R];
    #pragma unroll
    for (int r=0; r<R; ++r) {
        if (r >= rr) break;
        const int m = m0 + r;
        float post[4];
        #pragma unroll
        for (int i=0; i<4; ++i) post[i]=b.post[m*4+i];
        float* __restrict__ streams = b.streams + (size_t)m*4*4096;
        float sq=0.f;
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            const float y = block((size_t)(m-first)*4096 + h);
            #pragma unroll
            for (int i=0; i<4; ++i) {
                const float v=narrow(b, post[i]*y+mix_all[r][idx][i]);
                streams[i*4096+h]=v;
                mix_all[r][idx][i]=v;
                sq+=v*v;
            }
        }
        inv[r]=rsqrtf(sum_block(sq)/(4*4096)+eps);
        if (!threadIdx.x) b.inv[m]=inv[r];
    }
    #pragma unroll 1
    for (int pass=0; pass<3; ++pass) {
        float sum[R][8];
        #pragma unroll
        for (int r=0; r<R; ++r)
            #pragma unroll
            for (int c=0; c<8; ++c) sum[r][c]=0.f;
        const uint4* __restrict__ fp = fnp + (size_t)pass*64*THREADS + threadIdx.x;
        #pragma unroll
        for (int i=0; i<4; ++i) {
            #pragma unroll
            for (int idx=0; idx<16; ++idx) {
                const uint4 q = __ldg(fp + (i*16 + idx)*THREADS);
                const unsigned u[4] = {q.x, q.y, q.z, q.w};
                float f[8];
                #pragma unroll
                for (int k=0; k<4; ++k) { f[2*k] = __uint_as_float(u[k] << 16); f[2*k+1] = __uint_as_float(u[k] & 0xffff0000u); }
                #pragma unroll
                for (int r=0; r<R; ++r) {
                    const float si = mix_all[r][idx][i] * inv[r];
                    #pragma unroll
                    for (int c=0; c<8; ++c) sum[r][c] += si * f[c];
                }
            }
        }
        #pragma unroll
        for (int r=0; r<R; ++r)
            #pragma unroll
            for (int c=0; c<8; ++c) {
                float v = sum[r][c];
                for (int d=16; d; d/=2) v += __shfl_down_sync(0xffffffffu, v, d);
                if (!(threadIdx.x & 31)) sh_sums[r][c][threadIdx.x >> 5] = v;
            }
        __syncthreads();
        const int c = threadIdx.x >> 5, lane = threadIdx.x & 31;
        #pragma unroll
        for (int r=0; r<R; ++r) {
            if (r >= rr) break;
            float v = lane < 8 ? sh_sums[r][c][lane] : 0.f;
            for (int d=16; d; d/=2) v += __shfl_down_sync(0xffffffffu, v, d);
            if (!lane) {
                sh_mix[r][pass*8+c] = v;
                b.mix[(m0 + r)*24+pass*8+c] = v;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r=0; r<R; ++r) {
        if (r >= rr) break;
        const int m = m0 + r;
        coefficients_warp(sh_mix[r],w.base,w.scale,b.pre+4*m,b.post+4*m,b.comb+16*m,sh_pre[r]);
    }
    __syncthreads();
    #pragma unroll
    for (int r=0; r<R; ++r) {
        if (r >= rr) break;
        const int m = m0 + r;
        const float p0 = sh_pre[r][0], p1 = sh_pre[r][1], p2 = sh_pre[r][2], p3 = sh_pre[r][3];
        float csq = 0.f;
        float rv[16];
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            float val = 0.f;
            val += p0 * mix_all[r][idx][0];
            val += p1 * mix_all[r][idx][1];
            val += p2 * mix_all[r][idx][2];
            val += p3 * mix_all[r][idx][3];
            rv[idx] = val;
            csq += val * val;
        }
        const float rinv = rsqrtf(sum_block(csq) / 4096 + eps);
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            const float val = rv[idx] * rinv * __bfloat162float(__ldg(w.norm + h));
            if (out) out[(size_t)m * 4096 + h] = __float2half_rn(val);
            if (normalized) normalized[(size_t)m * 4096 + h] = val;
        }
    }
}
template<class Block, int R>
__global__ void __launch_bounds__(THREADS, 1) k_mhc_out_inv_prepare_pk(
    Buffers b, Block block, Weights w, const uint4* __restrict__ fnp, int M, float eps,
    __half* out, float* normalized=nullptr)
{
    mhc_out_inv_prepare_pk_body<Block, R>(b, block, w, fnp, 0, M, eps, out, normalized);
}
// owner-range entry (Flash SP, G6_PF_RS=2): rows [first, mend) of the full lane, grid ceil(rows / R).
template<class Block, int R>
__global__ void __launch_bounds__(THREADS, 1) k_mhc_out_inv_prepare_pk_sp(
    Buffers b, Block block, Weights w, const uint4* __restrict__ fnp, int first, int mend, float eps,
    __half* out, float* normalized)
{
    mhc_out_inv_prepare_pk_body<Block, R>(b, block, w, fnp, first, mend, eps, out, normalized);
}
// k_mhc_out_inv_prepare_pk without the register-resident rows: wait first, one pass writes the new streams, then
// the 24 fn sums (all 3 passes in one sweep, each sum still in (i, idx) order) and the collapse re-read the row
// from L2. Same values, same order per thread, same reduction trees -> bit-identical. Low register use lets MINB
// CTAs share an SM, so the DRAM phase of one row overlaps the L2/FMA phase of another. f16 / bf16: optional
// fragment tables (bind::Xfrag layout, rows 0..M) of the normalized rows, as k_mhc_fragments would build them.
// Row body shared by k_mhc_out_inv_prepare_rr and its owner-range (Flash SP) entry point, as _pk above: the
// range is global rows [first, mend), the block index owner-local, fragment padding uses the FULL lane M_lane
// (SP rejects G6_PF_XFRAG, so f16/bf16 are null there; the parameter keeps the entry points one shape).
template<class Block, int R, int MINB>
__device__ __forceinline__ void mhc_out_inv_prepare_rr_body(
    Buffers b, Block& block, Weights w, const uint4* __restrict__ fnp, int first, int mend, int M_lane, float eps,
    __half* out, float* normalized, unsigned* f16, unsigned* bf16) {
    __shared__ float sh_sums[R][24][8];
    __shared__ float sh_mix[R][24];
    __shared__ float sh_pre[R][4];
    const int m0 = first + blockIdx.x * R;
    const int rr = mend - m0 < R ? mend - m0 : R;   // owner-end guard, uniform over the CTA
    const unsigned long long step = block.init();
    block.wait(step);
    float inv[R];
    #pragma unroll
    for (int r=0; r<R; ++r) {
        inv[r] = 0.f;
        if (r >= rr) break;
        const int m = m0 + r;
        float comb[16], post[4];
        #pragma unroll
        for (int k=0; k<16; ++k) comb[k]=b.comb[m*16+k];
        #pragma unroll
        for (int i=0; i<4; ++i) post[i]=b.post[m*4+i];
        float* __restrict__ streams = b.streams + (size_t)m*4*4096;
        float sq=0.f;
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            float old[4];
            #pragma unroll
            for (int j=0; j<4; ++j) old[j]=__ldg(streams+j*4096+h);
            const float y = block((size_t)(m-first)*4096 + h);
            #pragma unroll
            for (int i=0; i<4; ++i) {
                float mix=0.f;
                #pragma unroll
                for (int j=0; j<4; ++j) mix+=comb[j*4+i]*old[j];
                const float v=narrow(b, post[i]*y+mix);
                streams[i*4096+h]=v;
                sq+=v*v;
            }
        }
        inv[r]=rsqrtf(sum_block(sq)/(4*4096)+eps);
        if (!threadIdx.x) b.inv[m]=inv[r];
    }
    float sum[R][24];
    #pragma unroll
    for (int r=0; r<R; ++r)
        #pragma unroll
        for (int c=0; c<24; ++c) sum[r][c]=0.f;
    const uint4* __restrict__ fp = fnp + threadIdx.x;
    constexpr int UNR = MHC_RR_UNROLL;
    #pragma unroll UNR
    for (int k=0; k<64; ++k) {   // k = i*16 + idx
        const int h = threadIdx.x + (k & 15)*THREADS + (k >> 4)*4096;
        float si[R];
        #pragma unroll
        for (int r=0; r<R; ++r) si[r] = r < rr ? __ldcg(b.streams + (size_t)(m0 + r)*4*4096 + h) * inv[r] : 0.f;
        #pragma unroll
        for (int pass=0; pass<3; ++pass) {
            const uint4 q = __ldg(fp + (pass*64 + k)*THREADS);
            const unsigned u[4] = {q.x, q.y, q.z, q.w};
            float f[8];
            #pragma unroll
            for (int e=0; e<4; ++e) { f[2*e] = __uint_as_float(u[e] << 16); f[2*e+1] = __uint_as_float(u[e] & 0xffff0000u); }
            #pragma unroll
            for (int r=0; r<R; ++r)
                #pragma unroll
                for (int c=0; c<8; ++c) sum[r][pass*8+c] += si[r] * f[c];
        }
    }
    #pragma unroll
    for (int r=0; r<R; ++r)
        #pragma unroll
        for (int c=0; c<24; ++c) {
            float v = sum[r][c];
            for (int d=16; d; d/=2) v += __shfl_down_sync(0xffffffffu, v, d);
            if (!(threadIdx.x & 31)) sh_sums[r][c][threadIdx.x >> 5] = v;
        }
    __syncthreads();
    {
        const int wc = threadIdx.x >> 5, lane = threadIdx.x & 31;
        #pragma unroll
        for (int r=0; r<R; ++r) {
            if (r >= rr) break;
            #pragma unroll
            for (int pass=0; pass<3; ++pass) {
                const int c = pass*8 + wc;
                float v = lane < 8 ? sh_sums[r][c][lane] : 0.f;
                for (int d=16; d; d/=2) v += __shfl_down_sync(0xffffffffu, v, d);
                if (!lane) {
                    sh_mix[r][c] = v;
                    b.mix[(m0 + r)*24+c] = v;
                }
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int r=0; r<R; ++r) {
        if (r >= rr) break;
        const int m = m0 + r;
        coefficients_warp(sh_mix[r],w.base,w.scale,b.pre+4*m,b.post+4*m,b.comb+16*m,sh_pre[r]);
    }
    __syncthreads();
    #pragma unroll
    for (int r=0; r<R; ++r) {
        if (r >= rr) break;
        const int m = m0 + r;
        const float* __restrict__ s = b.streams + (size_t)m*4*4096;
        const float p0 = sh_pre[r][0], p1 = sh_pre[r][1], p2 = sh_pre[r][2], p3 = sh_pre[r][3];
        float csq = 0.f;
        float rv[16];
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            float val = 0.f;
            val += p0 * __ldcg(s + h);
            val += p1 * __ldcg(s + 4096 + h);
            val += p2 * __ldcg(s + 8192 + h);
            val += p3 * __ldcg(s + 12288 + h);
            rv[idx] = val;
            csq += val * val;
        }
        const float rinv = rsqrtf(sum_block(csq) / 4096 + eps);
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            const float val = rv[idx] * rinv * __bfloat162float(__ldg(w.norm + h));
            if (out) out[(size_t)m * 4096 + h] = __float2half_rn(val);
            if (normalized) normalized[(size_t)m * 4096 + h] = val;
            if (f16 || bf16) {   // fragment tables of all M rows (zero pad rows after the last), as k_mhc_fragments
                const float b_val = __shfl_xor_sync(0xffffffffu, val, 1);
                if (!(threadIdx.x & 1)) write_frag_pair(m, M_lane, 4096, h, val, b_val, f16, bf16);
            }
        }
    }
}
template<class Block, int R, int MINB>
__global__ void __launch_bounds__(THREADS, MINB) k_mhc_out_inv_prepare_rr(
    Buffers b, Block block, Weights w, const uint4* __restrict__ fnp, int M, float eps,
    __half* out, float* normalized=nullptr, unsigned* f16=nullptr, unsigned* bf16=nullptr)
{
    mhc_out_inv_prepare_rr_body<Block, R, MINB>(b, block, w, fnp, 0, M, M, eps, out, normalized, f16, bf16);
}
// owner-range entry (Flash SP, G6_PF_RS=2): rows [first, mend) of the full lane, grid ceil(rows / R).
template<class Block, int R, int MINB>
__global__ void __launch_bounds__(THREADS, MINB) k_mhc_out_inv_prepare_rr_sp(
    Buffers b, Block block, Weights w, const uint4* __restrict__ fnp, int first, int mend, int M, float eps,
    __half* out, float* normalized, unsigned* f16, unsigned* bf16)
{
    mhc_out_inv_prepare_rr_body<Block, R, MINB>(b, block, w, fnp, first, mend, M, eps, out, normalized, f16, bf16);
}
template<class Block>
__global__ void __launch_bounds__(THREADS, 2) k_mhc_consume_fused(
    Buffers b, Block block, Weights w, int H, float eps,
    __half* out, float* normalized=nullptr, unsigned* f16=nullptr, unsigned* bf16=nullptr) {
    __shared__ float sh_pre[4];
    __shared__ int sh_last;
    const int o=blockIdx.x, m=blockIdx.y, M=gridDim.y;
    const unsigned long long step = block.init();
    const unsigned seq32 = block.seq32(step);
    float* __restrict__ s = b.streams + (size_t)m*4*H;
    const __nv_bfloat16* __restrict__ fn = w.fn + (size_t)o*4*H;
    asm volatile("prefetch.global.L1 [%0];" :: "l"(fn + threadIdx.x * 64));
    float inv;
    if (o == 0) {
        float post[4], comb[16];
        #pragma unroll
        for (int i=0; i<4; ++i) post[i]=b.post[m*4+i];
        #pragma unroll
        for (int k=0; k<16; ++k) comb[k]=b.comb[m*16+k];
        block.wait(step);
        float y_all[16];
        #pragma unroll
        for (int idx=0; idx<16; ++idx)
            y_all[idx] = block((size_t)m*4096 + threadIdx.x + idx*THREADS);
        float sq=0.f;
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            const float y = y_all[idx];
            float old[4];
            #pragma unroll
            for (int j=0; j<4; ++j) old[j]=__ldg(s+j*4096+h);
            #pragma unroll
            for (int i=0; i<4; ++i) {
                float mix=0.f;
                #pragma unroll
                for (int j=0; j<4; ++j) mix+=comb[j*4+i]*old[j];
                const float v=narrow(b, post[i]*y+mix);
                s[i*4096+h]=v; sq+=v*v;
            }
        }
        inv=rsqrtf(sum_block(sq)/(4*4096)+eps);
        if (!threadIdx.x) {
            b.inv[m]=inv;
            __threadfence();
            atomicExch(b.done+32+m, seq32);
        }
    } else {
        if (threadIdx.x == 0) {
            while (atomicAdd(b.done+32+m, 0u) != seq32) {}
        }
        __syncthreads();
        inv = __ldcg(b.inv+m);
    }
    float sum=0.f;
    #pragma unroll 16
    for (int idx=0; idx<64; ++idx) {
        const int k=threadIdx.x + idx*THREADS;
        sum += (__ldcg(s+k)*inv) * __bfloat162float(__ldg(fn+k));
    }
    const float v=sum_block(sum);
    if (!threadIdx.x) {
        b.mix[m*24+o]=v;
        __threadfence();
        sh_last = (atomicInc(b.done+m, MIX-1) == MIX-1);
    }
    if (m + 1 == M && (f16 || bf16)) {
        const int padded = (M + 7) / 8 * 8;
        const int mp = M + o;
        if (mp < padded) {
            const int ks = threadIdx.x;
            const size_t base = (size_t)(mp / 8) * 256 * 64 + (ks * 32 + (mp % 8) * 4) * 2;
            if (f16) {
                uint4* d = reinterpret_cast<uint4*>(f16 + base);
                d[0] = make_uint4(0u, 0u, 0u, 0u);
                d[1] = make_uint4(0u, 0u, 0u, 0u);
            }
            if (bf16) {
                uint4* d = reinterpret_cast<uint4*>(bf16 + base);
                d[0] = make_uint4(0u, 0u, 0u, 0u);
                d[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }
    __syncthreads();
    if (!sh_last) return;
    coefficients_warp(b.mix+24*m,w.base,w.scale,b.pre+4*m,b.post+4*m,b.comb+16*m,sh_pre);
    __syncthreads();
    const float p0 = sh_pre[0], p1 = sh_pre[1], p2 = sh_pre[2], p3 = sh_pre[3];
    float sq = 0.f;
    float rv[16];
    #pragma unroll
    for (int idx = 0; idx < 16; ++idx) {
        const int h = threadIdx.x + idx * THREADS;
        float val = 0.f;
        val += p0 * __ldcg(s + h);
        val += p1 * __ldcg(s + 4096 + h);
        val += p2 * __ldcg(s + 8192 + h);
        val += p3 * __ldcg(s + 12288 + h);
        rv[idx] = val;
        sq += val * val;
    }
    const float rinv = rsqrtf(sum_block(sq) / 4096 + eps);
    #pragma unroll
    for (int idx = 0; idx < 16; ++idx) {
        const int h = threadIdx.x + idx * THREADS;
        const float val = rv[idx] * rinv * __bfloat162float(__ldg(w.norm + h));
        if (out) out[(size_t)m * 4096 + h] = __float2half_rn(val);
        if (normalized) normalized[(size_t)m * 4096 + h] = val;
        if (f16 || bf16) {
            const float b_val = __shfl_xor_sync(0xffffffffu, val, 1);
            if (!(threadIdx.x & 1)) write_frag_pair_nozero(m, 4096, h, val, b_val, f16, bf16);
        }
    }
}
__global__ void __launch_bounds__(THREADS) k_mhc_final(Buffers b, const __nv_bfloat16* norm, int H, float eps,
                           __half* out, float* normalized=nullptr, float* mean=nullptr,
                           unsigned* f16=nullptr, unsigned* bf16=nullptr) {
    const int m=blockIdx.x, M=gridDim.x;
    const float* __restrict__ s = b.streams + (size_t)m*4*H;
    if (H == 4096) {
        float sq=0.f;
        float rv[16];
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            float v=0.f;
            #pragma unroll
            for (int j=0; j<4; ++j) v+=__ldg(s+j*4096+h);
            v*=.25f; rv[idx]=v; sq+=v*v;
            if (mean) mean[(size_t)m*4096+h]=v;
        }
        const float inv=rsqrtf(sum_block(sq)/4096+eps);
        #pragma unroll
        for (int idx=0; idx<16; ++idx) {
            const int h = threadIdx.x + idx*THREADS;
            const float v=rv[idx]*inv*__bfloat162float(__ldg(norm+h));
            if (out) out[(size_t)m*4096+h]=__float2half_rn(v);
            if (normalized) normalized[(size_t)m*4096+h]=v;
            if (f16 || bf16) {
                const float b_val = __shfl_xor_sync(0xffffffffu, v, 1);
                if (!(threadIdx.x & 1)) write_frag_pair(m, M, 4096, h, v, b_val, f16, bf16);
            }
        }
        return;
    }
    extern __shared__ float row[];
    float sq=0.f;
    for (int h=threadIdx.x; h<H; h+=THREADS) {
        float v=0.f;
        for (int j=0; j<4; ++j) v+=s[j*H+h];
        v*=.25f; row[h]=v; sq+=v*v;
        if (mean) mean[(size_t)m*H+h]=v;
    }
    const float inv=rsqrtf(sum_block(sq)/H+eps);
    for (int h=threadIdx.x; h<H; h+=THREADS) {
        const float v=row[h]*inv*__bfloat162float(norm[h]);
        if (out) out[(size_t)m*H+h]=__float2half_rn(v);
        if (normalized) normalized[(size_t)m*H+h]=v;
        if (f16 || bf16) {
            const float b_val = __shfl_xor_sync(__activemask(), v, 1);
            if (!(threadIdx.x & 1)) write_frag_pair(m, M, H, h, v, b_val, f16, bf16);
        }
    }
}
// row body shared by k_mhc_out_inv_final and its owner-range (Flash SP, G6_PF_RS=2) entry point: m is the GLOBAL
// lane row, first the first row of the launched range (block index k is owner-local: (m - first) * 4096 + h).
// M_lane is the FULL lane row count (write_frag_pair shape); OFF passes first = 0, M_lane = gridDim.x.
template<class Block>
__device__ __forceinline__ void mhc_out_inv_final_body(Buffers b, Block& block, const __nv_bfloat16* norm, int m, int first, int M_lane,
                                                       int H, float eps, __half* out, float* normalized, float* mean,
                                                       unsigned* f16, unsigned* bf16) {
    const unsigned long long step = block.init();
    float post[4], comb[16];
    #pragma unroll
    for (int i=0; i<4; ++i) post[i]=b.post[m*4+i];
    #pragma unroll
    for (int k=0; k<16; ++k) comb[k]=b.comb[m*16+k];
    float* __restrict__ streams = b.streams + (size_t)m*4*H;
    float mix_all[16][4];
    #pragma unroll
    for (int idx=0; idx<16; ++idx) {
        const int h = threadIdx.x + idx*THREADS;
        float old[4];
        #pragma unroll
        for (int j=0; j<4; ++j) old[j]=__ldg(streams+j*4096+h);
        #pragma unroll
        for (int i=0; i<4; ++i) {
            float mix=0.f;
            #pragma unroll
            for (int j=0; j<4; ++j) mix+=comb[j*4+i]*old[j];
            mix_all[idx][i]=mix;
        }
    }
    block.wait(step);
    float sq=0.f, msq=0.f;
    float rv[16];
    float y_all[16];
    #pragma unroll
    for (int idx=0; idx<16; ++idx)
        y_all[idx] = block((size_t)(m-first)*4096 + threadIdx.x + idx*THREADS);
    #pragma unroll
    for (int idx=0; idx<16; ++idx) {
        const int h = threadIdx.x + idx*THREADS;
        const float y = y_all[idx];
        float mv=0.f;
        #pragma unroll
        for (int i=0; i<4; ++i) {
            const float v=narrow(b, post[i]*y+mix_all[idx][i]);
            streams[i*4096+h]=v; sq+=v*v;
            mv+=v;
        }
        mv*=.25f; rv[idx]=mv; msq+=mv*mv;
        if (mean) mean[(size_t)m*4096+h]=mv;
    }
    const float inv=rsqrtf(sum_block(sq)/(4*4096)+eps);
    if (!threadIdx.x) b.inv[m]=inv;
    const float minv=rsqrtf(sum_block(msq)/4096+eps);
    #pragma unroll
    for (int idx=0; idx<16; ++idx) {
        const int h = threadIdx.x + idx*THREADS;
        const float v=rv[idx]*minv*__bfloat162float(__ldg(norm+h));
        if (out) out[(size_t)m*4096+h]=__float2half_rn(v);
        if (normalized) normalized[(size_t)m*4096+h]=v;
        if (f16 || bf16) {
            const float b_val = __shfl_xor_sync(0xffffffffu, v, 1);
            if (!(threadIdx.x & 1)) write_frag_pair(m, M_lane, 4096, h, v, b_val, f16, bf16);
        }
    }
}
template<class Block>
__global__ void __launch_bounds__(THREADS) k_mhc_out_inv_final(Buffers b, Block block, const __nv_bfloat16* norm, int H, float eps,
                                                               __half* out, float* normalized=nullptr, float* mean=nullptr,
                                                               unsigned* f16=nullptr, unsigned* bf16=nullptr) {
    mhc_out_inv_final_body(b, block, norm, blockIdx.x, 0, gridDim.x, H, eps, out, normalized, mean, f16, bf16);
}
// owner-range entry (Flash SP, G6_PF_RS=2): one CTA per GLOBAL row [first, mend); grid = mend - first (>= 1).
// Fragment tables are not built on the SP path (it rejects G6_PF_XFRAG); mean stays (the final residual rows).
// M_lane is only the write_frag_pair shape, and f16/bf16 are null here, so the owner row count (gridDim.x) is
// passed; it is NOT the full lane M. A fragment-building SP entry would have to take the full lane M.
template<class Block>
__global__ void __launch_bounds__(THREADS) k_mhc_out_inv_final_sp(Buffers b, Block block, const __nv_bfloat16* norm, int first, int H, float eps,
                                                                  __half* out, float* normalized, float* mean) {
    mhc_out_inv_final_body(b, block, norm, first + blockIdx.x, first, gridDim.x, H, eps, out, normalized, mean, nullptr, nullptr);
}
// Write both fragment tables directly from FP32 (do not round through FP16
// before BF16). Layout is bind::Xfrag: [tile8][K/16][32][2] words.
__global__ void k_mhc_fragments(const float* x, int M, int H, unsigned* f16, unsigned* bf16) {
    const int pair=blockIdx.x*blockDim.x+threadIdx.x;
    const int padded=(M+7)/8*8;
    if (pair>=padded*H/2) return;
    const int m=pair/(H/2), h=2*(pair%(H/2));
    const float a=m<M?x[(size_t)m*H+h]:0.f, b=m<M?x[(size_t)m*H+h+1]:0.f;
    const size_t dst=(size_t)(m/8)*(H/16)*64+((h/16)*32+(m%8)*4+(h%8)/2)*2+(h/8)%2;
    if (f16) f16[dst]=(unsigned)__half_as_ushort(__float2half_rn(a)) |
                     ((unsigned)__half_as_ushort(__float2half_rn(b))<<16);
    if (bf16) bf16[dst]=(unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(a)) |
                        ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(b))<<16);
}
inline void fragments(const float* x, int M, int H, unsigned* f16, unsigned* bf16, cudaStream_t st) {
    k_mhc_fragments<<<(((M+7)/8*8*H/2)+255)/256,256,0,st>>>(x,M,H,f16,bf16);
}
inline void prepare(Buffers b, Weights w, int M, int H, float eps, __half* out, cudaStream_t st,
                    float* normalized=nullptr, unsigned* f16=nullptr, unsigned* bf16=nullptr) {
    if (b.done) {
        k_mhc_prepare_fused<<<dim3(MIX,M),THREADS,H==4096?0:H*sizeof(float),st>>>(b,w,H,eps,out,normalized,f16,bf16);
    } else {
        k_mhc_mix<<<dim3(MIX,M),THREADS,0,st>>>(b,w,H);
        k_mhc_collapse_norm<<<M,THREADS,H*sizeof(float),st>>>(b,w,H,eps,out,nullptr,normalized,f16,bf16,true);
    }
}
// Owner-range fused prepare for the Flash SP path (G6_PF_RS=2): GLOBAL rows [first, mend) of a full M_lane-row
// lane. Fused path only (FlashPrefill always allocates hc_done); fragment tables are not built (SP rejects
// G6_PF_XFRAG). A zero-row owner launches nothing (a 0 grid is invalid) but still publishes its AG lane.
inline void prepare_range(Buffers b, Weights w, int first, int mend, int M_lane, int H, float eps,
                          __half* out, cudaStream_t st, float* normalized=nullptr) {
    if (!b.done) { fprintf(stderr, "g6: prepare_range: fused path only (hc_done == 0)\n"); exit(2); }
    const int rows = mend - first;
    if (rows < 1) return;
    k_mhc_prepare_fused_sp<<<dim3(MIX,rows),THREADS,H==4096?0:H*sizeof(float),st>>>(b,w,first,M_lane,H,eps,out,normalized);
}
} // namespace flash_mhc
