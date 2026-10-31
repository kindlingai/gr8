// mHC k_step split into two kernels for prefill row strips. Our own code (no ported code).
// An alternative to the single-launch k_step kernel (which launches 80 CTAs per 4-row tile, dumps 34.08 MB of
// uncoalesced strided partials to b.part, and runs __threadfence() + atomicInc on b.done before 1/80th of
// CTAs run the tail) with two right-sized split kernels operating on L2-resident row strips:
//   Stage 1 (k_decomp_mhc_post_mix_part):
//     - Grid: (ceil(M_strip / DECOMP_BM), NG), where DECOMP_BM = 8 rows/CTA (1 warp per row in Step A) and
//       NG = ceil((H/64) / DECOMP_KS) = 16 column groups (DECOMP_KS = 5 slices = 320 cols/CTA for H=5120).
//     - Putting row-tiles on blockIdx.x and column-group g on blockIdx.y schedules concurrent CTAs on the
//       same column group g so the 24.5 KB per-slice w.fn footprint hits L1/L2 across all active CTAs,
//       while DECOMP_BM = 8 halves w.fn loads per row vs RT = 4.
//     - Step A uses 1 warp per row (hh0 = lane * 2), vectorized 2-column loads/stores with .cs streaming
//       hints on b.streams and block so the 167.8 MB trunk does not evict y, b.part, or w.fn from L2.
//     - Shared memory tsm stores BF16 (4 KB for 8x256, exact because nw is rn16), and Step B loads each
//       row's tsm slice once as uint2 (8B) across all 3 q outputs, accumulating across DECOMP_KS = 5 slices
//       in registers before a single warp-shuffle reduction and a single 100% coalesced warp write to
//       b.part[((m * NG + g) * 32) + v]. Zero __threadfence(), zero atomicInc, zero b.done traffic.
//   Stage 2 (k_decomp_mhc_reduce_sinkhorn_norm):
//     - Grid: ceil(M_strip / DECOMP_BM), 256 threads (8 warps = 8 rows per CTA, 1 warp per row).
//     - Each warp independently reduces the NG = 16 coalesced partial vectors for its row m in registers
//       (zero shared memory reduction, zero __syncthreads() in the entire kernel), runs Sinkhorn coeff16
//       across all 8 warps in parallel (100% warp utilization), and rescales y[m, 0..H-1] with 16-byte
//       uint4 (8 BF16s) loads/stores while y_strip (5.24 MB for DECOMP_STRIP_M=512) is still hot in L2.
#pragma once
#include "ds_mhc.cuh"

namespace ds_mhc {

constexpr int DECOMP_BM = 8;
constexpr int DECOMP_PW = 64;
constexpr int DECOMP_KS = 5;
constexpr int DECOMP_PART_STRIDE = 32;
constexpr int DECOMP_STRIP_M = 512;

template <class B, class = void> struct decomp_has_dev_sum : std::false_type {};
template <class B>
struct decomp_has_dev_sum<B, std::void_t<decltype(std::declval<B>().sum)>>
    : std::is_same<decltype(std::declval<B>().sum), const __nv_bfloat16*> {};
template <class B> constexpr bool decomp_has_dev_sum_v = decomp_has_dev_sum<B>::value;

__device__ __forceinline__ void decomp_ld_t2_cs(const __nv_bfloat16* p, float& v0, float& v1) {
    unsigned u;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(u) : "l"(p));
    v0 = __uint_as_float(u << 16);
    v1 = __uint_as_float(u & 0xffff0000u);
}
__device__ __forceinline__ void decomp_ld_t2_cs(const float* p, float& v0, float& v1) {
    asm volatile("ld.global.cs.v2.f32 {%0, %1}, [%2];" : "=f"(v0), "=f"(v1) : "l"(p));
}
__device__ __forceinline__ void decomp_st_t2_cs(__nv_bfloat16* p, float v0, float v1) {
    const __nv_bfloat162 b2 = __floats2bfloat162_rn(v0, v1);
    const unsigned u = *reinterpret_cast<const unsigned*>(&b2);
    asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p), "r"(u) : "memory");
}
__device__ __forceinline__ void decomp_st_t2_cs(float* p, float v0, float v1) {
    asm volatile("st.global.cs.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(v0), "f"(v1) : "memory");
}

template <class TT, class WT, class Block, bool FUSE>
__global__ void __launch_bounds__(THREADS)
k_decomp_mhc_post_mix_part(BuffersT<TT> b, Weights<WT> w, Block block, int H, int m_base, int m_end, int NG, __nv_bfloat16* y) {
    __shared__ union {
        __align__(16) __nv_bfloat16 tsm[DECOMP_BM][4 * DECOMP_PW];
        float sh_out[DECOMP_BM][DECOMP_PART_STRIDE];
    } u_smem;
    __shared__ float sh_pre[DECOMP_BM][4];
    __shared__ float sh_post[DECOMP_BM][4];
    __shared__ float sh_comb[DECOMP_BM][16];

    const int NCH = H / DECOMP_PW;
    const int tile = blockIdx.x;
    const int g = blockIdx.y;
    const int m0 = m_base + tile * DECOMP_BM;
    const int nr = min(DECOMP_BM, m_end - m0);
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int wid = t >> 5;
    const int m = m0 + wid;
    const bool valid = (wid < nr);

    if constexpr (FUSE) {
        const unsigned long long step = block.init();
        block.wait(step);
    }

    // Load per-row pre / post / comb once per CTA into shared memory (broadcast to warp wid in Step A).
    if (wid < nr) {
        if (lane < 4) {
            sh_pre[wid][lane] = b.pre[(size_t)m * 4 + lane];
            if constexpr (FUSE) sh_post[wid][lane] = b.post[(size_t)m * 4 + lane];
        }
        if constexpr (FUSE) {
            if (lane < 16) sh_comb[wid][lane] = b.comb[(size_t)m * 16 + lane];
        }
    } else {
        // Zero invalid rows of tsm once so Step B needs no inner-loop row bounds branch.
        #pragma unroll
        for (int idx = lane * 2; idx < 4 * DECOMP_PW; idx += 64) {
            *reinterpret_cast<unsigned*>(&u_smem.tsm[wid][idx]) = 0u;
        }
    }
    __syncthreads();

    float d_acc[3][DECOMP_BM];
    #pragma unroll
    for (int q = 0; q < 3; ++q) {
        #pragma unroll
        for (int rr = 0; rr < DECOMP_BM; ++rr) d_acc[q][rr] = 0.f;
    }
    float sq_acc = 0.f, sqx_acc = 0.f;

    const int hh0 = lane * 2;
    const int j0 = lane >> 4;
    const int fn_hh0 = (lane & 15) * 4;
    const int e0 = lane * 4;

    #pragma unroll 1
    for (int s = 0; s < DECOMP_KS; ++s) {
        const int c = g * DECOMP_KS + s;
        if (c >= NCH) break;

        if (s > 0) __syncthreads();

        // Step A: warp wid processes row m = m0 + wid, column pair h0 = c*64 + lane*2.
        if (valid) {
            const int h0 = c * DECOMP_PW + hh0;
            TT* __restrict__ sm = b.streams + (size_t)m * 4 * H + h0;
            float xv0 = 0.f, xv1 = 0.f;
            if constexpr (FUSE) {
                float old0[4], old1[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    decomp_ld_t2_cs(sm + (size_t)j * H, old0[j], old1[j]);
                }
                float yv0, yv1;
                if constexpr (decomp_has_dev_sum_v<Block>) {
                    decomp_ld_t2_cs(block.sum + (size_t)m * H + h0, yv0, yv1);
                    yv0 = rn16(yv0);
                    yv1 = rn16(yv1);
                } else {
                    yv0 = rn16(block((size_t)m * H + h0));
                    yv1 = rn16(block((size_t)m * H + h0 + 1));
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float pi = sh_post[wid][i];
                    float mix0 = 0.f, mix1 = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float cj = sh_comb[wid][j * 4 + i];
                        mix0 += cj * old0[j];
                        mix1 += cj * old1[j];
                    }
                    const float nw0_i = rn16(pi * yv0 + mix0);
                    const float nw1_i = rn16(pi * yv1 + mix1);
                    decomp_st_t2_cs(sm + (size_t)i * H, nw0_i, nw1_i);
                    const float pri = sh_pre[wid][i];
                    *reinterpret_cast<__nv_bfloat162*>(&u_smem.tsm[wid][i * DECOMP_PW + hh0]) = __floats2bfloat162_rn(nw0_i, nw1_i);
                    sq_acc = fmaf(nw0_i, nw0_i, sq_acc);
                    sq_acc = fmaf(nw1_i, nw1_i, sq_acc);
                    xv0 += pri * nw0_i;
                    xv1 += pri * nw1_i;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float nw0_i, nw1_i;
                    decomp_ld_t2_cs(sm + (size_t)i * H, nw0_i, nw1_i);
                    const float pri = sh_pre[wid][i];
                    *reinterpret_cast<__nv_bfloat162*>(&u_smem.tsm[wid][i * DECOMP_PW + hh0]) = __floats2bfloat162_rn(nw0_i, nw1_i);
                    sq_acc = fmaf(nw0_i, nw0_i, sq_acc);
                    sq_acc = fmaf(nw1_i, nw1_i, sq_acc);
                    xv0 += pri * nw0_i;
                    xv1 += pri * nw1_i;
                }
            }
            xv0 = rn16(xv0);
            xv1 = rn16(xv1);
            sqx_acc = fmaf(xv0, xv0, sqx_acc);
            sqx_acc = fmaf(xv1, xv1, sqx_acc);
            *reinterpret_cast<__nv_bfloat162*>(y + (size_t)m * H + h0) = __floats2bfloat162_rn(xv0, xv1);
        }

        __syncthreads();

        // Step B: load fn slice for q = 0..2 and accumulate dot products across all DECOMP_BM = 8 rows.
        // Each row's BF16 tsm slice is loaded once (as two 8-byte uint2s) and reused across q = 0..2.
        float fa[3][4], fb[3][4];
        #pragma unroll
        for (int q = 0; q < 3; ++q) {
            const WT* fn = w.fn + (size_t)(wid + 8 * q) * 4 * H + (size_t)c * DECOMP_PW + fn_hh0;
            ld4(fn + (size_t)j0 * H, fa[q]);
            ld4(fn + (size_t)(j0 + 2) * H, fb[q]);
        }

        #pragma unroll
        for (int rr = 0; rr < DECOMP_BM; ++rr) {
            const uint2 u0 = *reinterpret_cast<const uint2*>(&u_smem.tsm[rr][e0]);
            const uint2 u1 = *reinterpret_cast<const uint2*>(&u_smem.tsm[rr][128 + e0]);
            const float a0x = __uint_as_float(u0.x << 16);
            const float a0y = __uint_as_float(u0.x & 0xffff0000u);
            const float a0z = __uint_as_float(u0.y << 16);
            const float a0w = __uint_as_float(u0.y & 0xffff0000u);
            const float a1x = __uint_as_float(u1.x << 16);
            const float a1y = __uint_as_float(u1.x & 0xffff0000u);
            const float a1z = __uint_as_float(u1.y << 16);
            const float a1w = __uint_as_float(u1.y & 0xffff0000u);
            #pragma unroll
            for (int q = 0; q < 3; ++q) {
                float v = d_acc[q][rr];
                v = fmaf(a0x, fa[q][0], v);
                v = fmaf(a0y, fa[q][1], v);
                v = fmaf(a0z, fa[q][2], v);
                v = fmaf(a0w, fa[q][3], v);
                v = fmaf(a1x, fb[q][0], v);
                v = fmaf(a1y, fb[q][1], v);
                v = fmaf(a1z, fb[q][2], v);
                v = fmaf(a1w, fb[q][3], v);
                d_acc[q][rr] = v;
            }
        }
    }

    // Single warp-shuffle reduction per CTA after accumulating all DECOMP_KS slices.
    for (int d = 16; d; d /= 2) {
        sq_acc += __shfl_down_sync(0xffffffffu, sq_acc, d);
        sqx_acc += __shfl_down_sync(0xffffffffu, sqx_acc, d);
    }
    #pragma unroll
    for (int q = 0; q < 3; ++q) {
        #pragma unroll
        for (int rr = 0; rr < DECOMP_BM; ++rr) {
            for (int d = 16; d; d /= 2) {
                d_acc[q][rr] += __shfl_down_sync(0xffffffffu, d_acc[q][rr], d);
            }
        }
    }

    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int q = 0; q < 3; ++q) {
            #pragma unroll
            for (int rr = 0; rr < DECOMP_BM; ++rr) {
                u_smem.sh_out[rr][wid + 8 * q] = d_acc[q][rr];
            }
        }
        u_smem.sh_out[wid][MIX] = sq_acc;
        u_smem.sh_out[wid][MIX + 1] = sqx_acc;
    }

    __syncthreads();

    // Coalesced single-warp write per valid row to b.part[((m * NG + g) * 32) + lane].
    if (valid && lane < NV) {
        b.part[((size_t)m * NG + g) * DECOMP_PART_STRIDE + lane] = u_smem.sh_out[wid][lane];
    }
}

template <class TT, class WT>
__global__ void __launch_bounds__(THREADS)
k_decomp_mhc_reduce_sinkhorn_norm(BuffersT<TT> b, Weights<WT> w, int H, int m_base, int m_end, int NG, float eps, __nv_bfloat16* y) {
    __shared__ float mixs[DECOMP_BM][MIX];

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = m_base + blockIdx.x * DECOMP_BM + wid;
    if (m >= m_end) return;

    // Step 2A: Warp wid reduces the NG = 16 coalesced partial vectors for row m in registers.
    const float* __restrict__ pm = b.part + (size_t)m * NG * DECOMP_PART_STRIDE + lane;
    float acc = 0.f;
    if (lane < NV) {
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            if (g < NG) acc += pm[g * DECOMP_PART_STRIDE];
        }
        for (int g = 16; g < NG; ++g) {
            acc += pm[g * DECOMP_PART_STRIDE];
        }
    }

    // Step 2B: Compute mix[0..23], inv, and Sinkhorn coeff16 in parallel across all 8 warps.
    const float sq_t = __shfl_sync(0xffffffffu, acc, MIX);
    const float sq_x = __shfl_sync(0xffffffffu, acc, MIX + 1);
    const float rs = rsqrtf(sq_t / (4 * H) + eps);
    const float inv = rsqrtf(sq_x / H + eps);

    if (lane < MIX) {
        const float mv = acc * rs;
        mixs[wid][lane] = mv;
        b.mix[(size_t)m * MIX + lane] = mv;
    }
    __syncwarp();
    coeff16(mixs[wid], w.base, w.scale, b.pre + (size_t)m * 4, b.post + (size_t)m * 4, b.comb + (size_t)m * 16);

    // Step 2C: Rescale y[m, 0..H-1] using 16-byte uint4 (8 BF16s) loads and stores.
    if ((H & 7) == 0) {
        uint4* __restrict__ y8 = reinterpret_cast<uint4*>(y + (size_t)m * H);
        const uint4* __restrict__ n8 = reinterpret_cast<const uint4*>(w.norm);
        const int n_vec = H >> 3;
        #pragma unroll 4
        for (int idx = lane; idx < n_vec; idx += 32) {
            const uint4 yv = y8[idx];
            const uint4 nv = __ldg(n8 + idx);
            uint4 out_v;
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(&yv);
            const __nv_bfloat162* n2 = reinterpret_cast<const __nv_bfloat162*>(&nv);
            __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out_v);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 yf = __bfloat1622float2(y2[k]);
                const float2 nf = __bfloat1622float2(n2[k]);
                o2[k] = __floats2bfloat162_rn(yf.x * inv * nf.x, yf.y * inv * nf.y);
            }
            y8[idx] = out_v;
        }
    } else {
        __nv_bfloat162* __restrict__ y2 = reinterpret_cast<__nv_bfloat162*>(y + (size_t)m * H);
        const __nv_bfloat162* __restrict__ n2 = reinterpret_cast<const __nv_bfloat162*>(w.norm);
        for (int p = lane; p < H / 2; p += 32) {
            const float2 yf = __bfloat1622float2(y2[p]);
            const float2 nf = __bfloat1622float2(__ldg(n2 + p));
            y2[p] = __floats2bfloat162_rn(yf.x * inv * nf.x, yf.y * inv * nf.y);
        }
    }
}

template <class WT, class TT>
inline void prepare_decomp(BuffersT<TT> b, Weights<WT> w, int M, int H, float eps, __nv_bfloat16* y, cudaStream_t st, int strip_m = DECOMP_STRIP_M, unsigned long long* kernel_launches = nullptr) {
    check_h(H);
    const int NCH = H / DECOMP_PW;
    const int NG = (NCH + DECOMP_KS - 1) / DECOMP_KS;
    for (int m_base = 0; m_base < M; m_base += strip_m) {
        const int m_end = m_base + strip_m < M ? m_base + strip_m : M;
        const int n_tiles = (m_end - m_base + DECOMP_BM - 1) / DECOMP_BM;
        k_decomp_mhc_post_mix_part<TT, WT, NoBlock, false><<<dim3(n_tiles, NG), THREADS, 0, st>>>(b, w, NoBlock{}, H, m_base, m_end, NG, y);
        if (kernel_launches) ++*kernel_launches;
        k_decomp_mhc_reduce_sinkhorn_norm<TT, WT><<<n_tiles, THREADS, 0, st>>>(b, w, H, m_base, m_end, NG, eps, y);
        if (kernel_launches) ++*kernel_launches;
    }
}

template <class WT, class Block, class TT>
inline void consume_prepare_decomp(BuffersT<TT> b, Block block, Weights<WT> w, int M, int H, float eps, __nv_bfloat16* y, cudaStream_t st, int strip_m = DECOMP_STRIP_M, unsigned long long* kernel_launches = nullptr) {
    check_h(H);
    const int NCH = H / DECOMP_PW;
    const int NG = (NCH + DECOMP_KS - 1) / DECOMP_KS;
    for (int m_base = 0; m_base < M; m_base += strip_m) {
        const int m_end = m_base + strip_m < M ? m_base + strip_m : M;
        const int n_tiles = (m_end - m_base + DECOMP_BM - 1) / DECOMP_BM;
        k_decomp_mhc_post_mix_part<TT, WT, Block, true><<<dim3(n_tiles, NG), THREADS, 0, st>>>(b, w, block, H, m_base, m_end, NG, y);
        if (kernel_launches) ++*kernel_launches;
        k_decomp_mhc_reduce_sinkhorn_norm<TT, WT><<<n_tiles, THREADS, 0, st>>>(b, w, H, m_base, m_end, NG, eps, y);
        if (kernel_launches) ++*kernel_launches;
    }
}

} // namespace ds_mhc
