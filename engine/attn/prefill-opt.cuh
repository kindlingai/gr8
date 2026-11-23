#pragma once
// Bit-exact (0-ULP) optimized prefill attention kernels gated by GR8_PF_ATTN_OPT (default 0):
//   - expand_index_rows_opt : vectorized E4M3->FP16 index key expand (18 regs, 0 spill)
//   - idx_score_ptx_n64     : PTX m16n8k16 N=64 indexer score, XOR-swizzled B, per-warp O,
//                             0 row-loop barriers, 2 CTAs/SM (128 regs, 0 spill)
//   - selopt2::select_rows_v2<256> : 1-pass sample bound + FP32 fast-reject filter + smem
//                             candidates + CUB BlockRadixSort, 2 CTAs/SM (86 regs, 0 spill)
//   - prepass_opt<FOLD>     : vectorized INT4 KV -> FP16 [576] table prepass (37 regs, 0 spill)
// Takes precedence over GR8_PF_ISGRT_CLEAN when both are enabled. QO core_b_half is untouched.

namespace pf_attn_opt {

using ga::half;
using ga::bf16;

__device__ __forceinline__ uint32_t saddr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mma16816(
    float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t a) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a)
        : "memory");
}

// ============================================================================
// 1. Vectorized E4M3 -> FP16 index key expand (matches gp::expand_index_rows)
// ============================================================================
__global__ void __launch_bounds__(256) expand_index_rows_opt(
    G6Attn fixed, half* __restrict__ ikey, float* __restrict__ scales, int count)
{
    auto a = ga::resolve(fixed);
    const int pos = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (pos >= count) return;
    float s = 0.f;
    const uint8_t* k = pos < a.context ? ga::ik_at(a, pos, s) : nullptr;
    uint2 out_v = make_uint2(0, 0);
    if (k) {
        const uint32_t raw = reinterpret_cast<const uint32_t*>(k)[lane];
        const __half2 h0 = __half2(__nv_cvt_fp8x2_to_halfraw2((uint16_t)(raw & 0xffffu), __NV_E4M3));
        const __half2 h1 = __half2(__nv_cvt_fp8x2_to_halfraw2((uint16_t)(raw >> 16), __NV_E4M3));
        out_v.x = *reinterpret_cast<const uint32_t*>(&h0);
        out_v.y = *reinterpret_cast<const uint32_t*>(&h1);
    }
    reinterpret_cast<uint2*>(ikey + (int64_t)pos * 128)[lane] = out_v;
    if (lane == 0) scales[pos] = s;
}

// ============================================================================
// 2. PTX m16n8k16 N=64 indexer score (matches gp::index_scores_rg8 bit-exact)
// ============================================================================
struct alignas(32) IdxSmem64 {
    half B[64 * 128];
    float2 O[8][32][16];
    float scales[64];
};
static_assert(sizeof(IdxSmem64) <= 50688, "IdxSmem64 fits 2 CTAs/SM");

__global__ void __launch_bounds__(256, 2) idx_score_ptx_n64(
    G6Attn a, int start, int n, int RG,
    const half* __restrict__ q, const float* __restrict__ weight,
    const half* __restrict__ keys, const float* __restrict__ scales,
    float* __restrict__ scores)
{
    extern __shared__ __align__(32) unsigned char raw_smem[];
    IdxSmem64& sm = *reinterpret_cast<IdxSmem64*>(raw_smem);
    const int rbeg = blockIdx.y * RG, rend = min(n, rbeg + RG);
    const int t0 = blockIdx.x * 64;
    const int t = threadIdx.x, w = t >> 5, l = t & 31, g = l >> 2, c = l & 3;

    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int i = t + u * 256;
        const int krow = i >> 4, ch = i & 15, key = t0 + krow;
        const int4 v = (key < a.context)
            ? __ldg(reinterpret_cast<const int4*>(keys + (int64_t)key * 128 + ch * 8))
            : make_int4(0, 0, 0, 0);
        *reinterpret_cast<int4*>(sm.B + (krow * 16 + (ch ^ (krow & 7))) * 8) = v;
    }
    if (t < 64) {
        const int key = t0 + t;
        sm.scales[t] = (key < a.context) ? __ldg(scales + key) : 0.f;
    }
    __syncthreads();

    const uint32_t b_base = saddr(sm.B);
    const int b_krow_lo = (l & 7) + ((l >> 4) & 1) * 8;
    const int b_ch_bit = (l >> 3) & 1, b_xor = l & 7;

    for (int row = rbeg + w; row < rend; row += 8) {
        const int pos = __ldg(a.positions + start + row);
        float* sc_row = scores + (int64_t)row * a.context + t0;
        if (t0 > pos) {
            if (t0 + l < a.context) sc_row[l] = -INFINITY;
            if (t0 + 32 + l < a.context) sc_row[32 + l] = -INFINITY;
            continue;
        }
        const float w_mine = __ldg(weight + (int64_t)row * 32 + l);
        const half* q0_base = q + ((int64_t)row * 32) * 128;
        const half* q1_base = q0_base + 16 * 128;

        float C0[8][4], C1[8][4];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            C0[j][0] = C0[j][1] = C0[j][2] = C0[j][3] = 0.f;
            C1[j][0] = C1[j][1] = C1[j][2] = C1[j][3] = 0.f;
        }
        #pragma unroll
        for (int k = 0; k < 128; k += 16) {
            const uint32_t a0_0 = __ldg(reinterpret_cast<const uint32_t*>(q0_base + g * 128 + k + 2 * c));
            const uint32_t a0_1 = __ldg(reinterpret_cast<const uint32_t*>(q0_base + (g + 8) * 128 + k + 2 * c));
            const uint32_t a0_2 = __ldg(reinterpret_cast<const uint32_t*>(q0_base + g * 128 + k + 2 * c + 8));
            const uint32_t a0_3 = __ldg(reinterpret_cast<const uint32_t*>(q0_base + (g + 8) * 128 + k + 2 * c + 8));
            const uint32_t a1_0 = __ldg(reinterpret_cast<const uint32_t*>(q1_base + g * 128 + k + 2 * c));
            const uint32_t a1_1 = __ldg(reinterpret_cast<const uint32_t*>(q1_base + (g + 8) * 128 + k + 2 * c));
            const uint32_t a1_2 = __ldg(reinterpret_cast<const uint32_t*>(q1_base + g * 128 + k + 2 * c + 8));
            const uint32_t a1_3 = __ldg(reinterpret_cast<const uint32_t*>(q1_base + (g + 8) * 128 + k + 2 * c + 8));
            const int ch_sw = ((k >> 3) + b_ch_bit) ^ b_xor;
            #pragma unroll
            for (int jp = 0; jp < 4; ++jp) {
                uint32_t b[4];
                ldsm4(b, b_base + (((jp * 16 + b_krow_lo) * 16) + ch_sw) * 16u);
                mma16816(C0[2 * jp],     a0_0, a0_1, a0_2, a0_3, b[0], b[1]);
                mma16816(C1[2 * jp],     a1_0, a1_1, a1_2, a1_3, b[0], b[1]);
                mma16816(C0[2 * jp + 1], a0_0, a0_1, a0_2, a0_3, b[2], b[3]);
                mma16816(C1[2 * jp + 1], a1_0, a1_1, a1_2, a1_3, b[2], b[3]);
            }
        }
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col2 = (j * 4 + c) ^ g, jj = hf * 4 + j;
                sm.O[w][g][col2]      = make_float2(C0[jj][0], C0[jj][1]);
                sm.O[w][g + 8][col2]  = make_float2(C0[jj][2], C0[jj][3]);
                sm.O[w][g + 16][col2] = make_float2(C1[jj][0], C1[jj][1]);
                sm.O[w][g + 24][col2] = make_float2(C1[jj][2], C1[jj][3]);
            }
            __syncwarp();
            const int k_local = hf * 32 + l, key = t0 + k_local, col2 = l >> 1, sub = l & 1;
            float x = 0.f;
            #pragma unroll
            for (int h = 0; h < 32; ++h) {
                const float wh = __shfl_sync(0xffffffffu, w_mine, h);
                const float2 fv = sm.O[w][h][col2 ^ (h & 7)];
                x += fmaxf(sub ? fv.y : fv.x, 0.f) * wh;
            }
            if (key < a.context) {
                x *= sm.scales[k_local];
                if (key > pos) x = -INFINITY;
                else if (!isfinite(x)) { ga::error(a, 4); x = -INFINITY; }
                sc_row[k_local] = x;
            }
            __syncwarp();
        }
    }
}

static inline cudaError_t launch_idx_n64(
    G6Attn a, int start, int n,
    const half* q, const float* weight, const half* keys, const float* scales,
    float* scores, cudaStream_t st)
{
    if (n <= 0 || a.context <= 0) return cudaSuccess;
    static const bool once =
        cudaFuncSetAttribute(idx_score_ptx_n64, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(IdxSmem64)) == cudaSuccess;
    (void)once;
    const int RG = (a.context >= 65536) ? 128 : 64;
    idx_score_ptx_n64<<<dim3((a.context + 63) / 64, (n + RG - 1) / RG), 256, sizeof(IdxSmem64), st>>>(
        a, start, n, RG, q, weight, keys, scales, scores);
    return cudaGetLastError();
}

// ============================================================================
// 2b. Native E4M3 x E4M3 mma.m16n8k32 N=64 indexer score (direct paged E4M3)
// ============================================================================
__device__ __forceinline__ void mma16832_e4(
    float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t f2_e4m3x2_idx(float a, float b) {
    uint32_t r;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; cvt.u32.u16 %0, t; }" : "=r"(r) : "f"(a), "f"(b));
    return r;
}

// Quantize q [n, 32, 128] FP16 -> q_e4 [n, 32, 128] E4M3 + w_eff [n, 32] = weight * (amax / 448).
// Since q_scale >= 0, ReLU(q_scale * dot) * weight == ReLU(dot) * (q_scale * weight) identically.
__global__ void __launch_bounds__(128) quant_idx_q_e4m3(
    int n, const half* __restrict__ q, const float* __restrict__ weight,
    uint8_t* __restrict__ q_e4, float* __restrict__ w_eff)
{
    const int row = blockIdx.x, h = blockIdx.y * 4 + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (row >= n || h >= 32) return;
    const half* src = q + ((int64_t)row * 32 + h) * 128;
    const uint2 raw = *reinterpret_cast<const uint2*>(src + l * 4);
    const __half2 h0 = *reinterpret_cast<const __half2*>(&raw.x);
    const __half2 h1 = *reinterpret_cast<const __half2*>(&raw.y);
    const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
    float am = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float qscale = am * (1.f / 448.f);
    const float inv = qscale > 0.f ? (448.f / am) : 0.f;
    const uint32_t p0 = f2_e4m3x2_idx(f0.x * inv, f0.y * inv);
    const uint32_t p1 = f2_e4m3x2_idx(f1.x * inv, f1.y * inv);
    reinterpret_cast<uint32_t*>(q_e4 + ((int64_t)row * 32 + h) * 128)[l] = p0 | (p1 << 16);
    if (l == 0) w_eff[(int64_t)row * 32 + h] = weight[(int64_t)row * 32 + h] * qscale;
}

struct alignas(32) IdxSmem64E4 {
    uint8_t B[64 * 128];   // 8 KB (half of FP16!)
    float2 O[8][32][16];   // 32 KB
    float scales[64];      // 256 B
};
static_assert(sizeof(IdxSmem64E4) <= 45056, "IdxSmem64E4 fits 2 CTAs/SM");

__global__ void __launch_bounds__(256, 2) idx_score_ptx_n64_e4m3(
    G6Attn fixed, int start, int n, int RG,
    const uint8_t* __restrict__ q_e4, const float* __restrict__ w_eff,
    float* __restrict__ scores)
{
    extern __shared__ __align__(32) unsigned char raw_smem[];
    IdxSmem64E4& sm = *reinterpret_cast<IdxSmem64E4*>(raw_smem);
    auto a = ga::resolve(fixed);
    const int rbeg = blockIdx.y * RG, rend = min(n, rbeg + RG);
    const int t0 = blockIdx.x * 64;
    const int t = threadIdx.x, w = t >> 5, l = t & 31, g = l >> 2, c = l & 3;

    const int live = min(64, max(0, a.context - t0));
    float s_dummy = 0.f;
    const uint8_t* src = live > 0 ? ga::ik_at(a, t0, s_dummy) : nullptr;
    const float* sc_ptr = src
        ? reinterpret_cast<const float*>(src - (t0 % a.page_size) * 128 + a.page_size * 128) + (t0 % a.page_size)
        : nullptr;

    #pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int i = t + u * 256;
        const int krow = i >> 3, ch = i & 7;
        const int4 v = (src && krow < live)
            ? __ldg(reinterpret_cast<const int4*>(src + (int64_t)krow * 128 + ch * 16))
            : make_int4(0, 0, 0, 0);
        *reinterpret_cast<int4*>(sm.B + (krow * 8 + (ch ^ (krow & 7))) * 16) = v;
    }
    if (t < 64) {
        sm.scales[t] = (sc_ptr && t < live) ? __ldg(sc_ptr + t) : 0.f;
    }
    __syncthreads();

    const uint32_t b_base = saddr(sm.B);
    const int b_krow_lo = (l & 7) + ((l >> 4) & 1) * 8;
    const int b_ch_bit = (l >> 3) & 1, b_xor = l & 7;

    for (int row = rbeg + w; row < rend; row += 8) {
        const int pos = __ldg(a.positions + start + row);
        float* sc_row = scores + (int64_t)row * a.context + t0;
        if (t0 > pos) {
            if (t0 + l < a.context) sc_row[l] = -INFINITY;
            if (t0 + 32 + l < a.context) sc_row[32 + l] = -INFINITY;
            continue;
        }
        const float w_mine = __ldg(w_eff + (int64_t)row * 32 + l);
        const uint8_t* q0_base = q_e4 + ((int64_t)row * 32) * 128;
        const uint8_t* q1_base = q0_base + 16 * 128;

        float C0[8][4], C1[8][4];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            C0[j][0] = C0[j][1] = C0[j][2] = C0[j][3] = 0.f;
            C1[j][0] = C1[j][1] = C1[j][2] = C1[j][3] = 0.f;
        }
        #pragma unroll
        for (int k = 0; k < 128; k += 32) {
            const uint32_t a0_0 = __ldg(reinterpret_cast<const uint32_t*>(q0_base + g * 128 + k + 4 * c));
            const uint32_t a0_1 = __ldg(reinterpret_cast<const uint32_t*>(q0_base + (g + 8) * 128 + k + 4 * c));
            const uint32_t a0_2 = __ldg(reinterpret_cast<const uint32_t*>(q0_base + g * 128 + k + 4 * c + 16));
            const uint32_t a0_3 = __ldg(reinterpret_cast<const uint32_t*>(q0_base + (g + 8) * 128 + k + 4 * c + 16));
            const uint32_t a1_0 = __ldg(reinterpret_cast<const uint32_t*>(q1_base + g * 128 + k + 4 * c));
            const uint32_t a1_1 = __ldg(reinterpret_cast<const uint32_t*>(q1_base + (g + 8) * 128 + k + 4 * c));
            const uint32_t a1_2 = __ldg(reinterpret_cast<const uint32_t*>(q1_base + g * 128 + k + 4 * c + 16));
            const uint32_t a1_3 = __ldg(reinterpret_cast<const uint32_t*>(q1_base + (g + 8) * 128 + k + 4 * c + 16));
            const int ch_sw = ((k >> 4) + b_ch_bit) ^ b_xor;
            #pragma unroll
            for (int jp = 0; jp < 4; ++jp) {
                uint32_t b[4];
                ldsm4(b, b_base + (((jp * 16 + b_krow_lo) * 8) + ch_sw) * 16u);
                mma16832_e4(C0[2 * jp],     a0_0, a0_1, a0_2, a0_3, b[0], b[1]);
                mma16832_e4(C1[2 * jp],     a1_0, a1_1, a1_2, a1_3, b[0], b[1]);
                mma16832_e4(C0[2 * jp + 1], a0_0, a0_1, a0_2, a0_3, b[2], b[3]);
                mma16832_e4(C1[2 * jp + 1], a1_0, a1_1, a1_2, a1_3, b[2], b[3]);
            }
        }
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col2 = (j * 4 + c) ^ g, jj = hf * 4 + j;
                sm.O[w][g][col2]      = make_float2(C0[jj][0], C0[jj][1]);
                sm.O[w][g + 8][col2]  = make_float2(C0[jj][2], C0[jj][3]);
                sm.O[w][g + 16][col2] = make_float2(C1[jj][0], C1[jj][1]);
                sm.O[w][g + 24][col2] = make_float2(C1[jj][2], C1[jj][3]);
            }
            __syncwarp();
            const int k_local = hf * 32 + l, key = t0 + k_local, col2 = l >> 1, sub = l & 1;
            float x = 0.f;
            #pragma unroll
            for (int h = 0; h < 32; ++h) {
                const float wh = __shfl_sync(0xffffffffu, w_mine, h);
                const float2 fv = sm.O[w][h][col2 ^ (h & 7)];
                x += fmaxf(sub ? fv.y : fv.x, 0.f) * wh;
            }
            if (key < a.context) {
                x *= sm.scales[k_local];
                if (key > pos) x = -INFINITY;
                else if (!isfinite(x)) { ga::error(a, 4); x = -INFINITY; }
                sc_row[k_local] = x;
            }
            __syncwarp();
        }
    }
}

static inline cudaError_t launch_idx_n64_e4m3(
    G6Attn a, int start, int n,
    const half* q, const float* weight,
    uint8_t* q_e4_buf, float* w_eff_buf,
    float* scores, cudaStream_t st)
{
    if (n <= 0 || a.context <= 0) return cudaSuccess;
    static const bool once =
        cudaFuncSetAttribute(idx_score_ptx_n64_e4m3, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(IdxSmem64E4)) == cudaSuccess;
    (void)once;
    quant_idx_q_e4m3<<<dim3(n, 8), 128, 0, st>>>(n, q, weight, q_e4_buf, w_eff_buf);
    const int RG = (a.context >= 65536) ? 128 : 64;
    idx_score_ptx_n64_e4m3<<<dim3((a.context + 63) / 64, (n + RG - 1) / RG), 256, sizeof(IdxSmem64E4), st>>>(
        a, start, n, RG, q_e4_buf, w_eff_buf, scores);
    return cudaGetLastError();
}

// ============================================================================
// 3. Indexer Select v2: FP32 fast-reject + 1-pass sample bound + smem cand + CUB sort
// ============================================================================
namespace selopt2 {

constexpr int TOP = ga::TOP, W = 11, NB = 1 << W;
constexpr int CAP_SMEM = 3072;

__device__ __forceinline__ uint32_t ord(float s) {
    uint32_t b = __float_as_uint(s);
    if ((b & 0x7fffffffu) >= 0x7f800000u) return 0;
    if ((b & 0x7fffffffu) == 0) b = 0;
    return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}

__device__ __forceinline__ float inv_ord(uint32_t o) {
    if (o <= 0x00800000u) return -3.4028234663852886e+38f;
    if (o >= 0xff800000u) return  3.4028234663852886e+38f;
    const uint32_t b = (o & 0x80000000u) ? (o ^ 0x80000000u) : ~o;
    return __uint_as_float(b);
}

__device__ __forceinline__ uint64_t make_key(float s, int id) {
    const uint32_t o = ord(s);
    return o ? ((uint64_t)o << 32) | (0xffffffffu - (uint32_t)id) : 0ull;
}

struct St { uint64_t pp, p; int lp, l, need, fin, nout, ncand, B, above, hb, total; };

template<int BT>
struct SmemSel {
    uint64_t out[TOP];
    uint64_t cand[CAP_SMEM];
    int hist[NB];
    typename cub::BlockScan<int, BT, cub::BLOCK_SCAN_WARP_SCANS>::TempStorage scan;
    St s;
};

template<int BT>
union Smem {
    SmemSel<BT> sel;
    typename cub::BlockRadixSort<uint64_t, BT, TOP / BT>::TempStorage sort;
};
static_assert(sizeof(Smem<256>) <= 50688, "selopt2<256> fits 2 CTAs/SM");

__device__ __forceinline__ void load_row4(const float* x, int len, bool v4, int i0, uint64_t* k) {
    if (v4 && i0 + 3 < len) {
        const float4 q = __ldg(reinterpret_cast<const float4*>(x + i0));
        k[0] = make_key(q.x, i0); k[1] = make_key(q.y, i0 + 1);
        k[2] = make_key(q.z, i0 + 2); k[3] = make_key(q.w, i0 + 3);
    } else {
        #pragma unroll
        for (int e = 0; e < 4; ++e) k[e] = (i0 + e < len) ? make_key(__ldg(x + i0 + e), i0 + e) : 0ull;
    }
}

__device__ __forceinline__ void append_out(bool take, uint64_t k, int* cnt, uint64_t* buf) {
    const unsigned m = __ballot_sync(0xffffffffu, take);
    if (!m) return;
    const int lane = threadIdx.x & 31, lead = __ffs(m) - 1;
    int base = 0;
    if (lane == lead) base = atomicAdd(cnt, __popc(m));
    base = __shfl_sync(0xffffffffu, base, lead);
    if (take) { const int slot = base + __popc(m & ((1u << lane) - 1)); if (slot < TOP) buf[slot] = k; }
}

__device__ __forceinline__ void append_cand(bool take, uint64_t k, int* cnt, uint64_t* sc, uint64_t* gc, int cap_tot) {
    const unsigned m = __ballot_sync(0xffffffffu, take);
    if (!m) return;
    const int lane = threadIdx.x & 31, lead = __ffs(m) - 1;
    int base = 0;
    if (lane == lead) base = atomicAdd(cnt, __popc(m));
    base = __shfl_sync(0xffffffffu, base, lead);
    if (take) {
        const int slot = base + __popc(m & ((1u << lane) - 1));
        if (slot < CAP_SMEM) sc[slot] = k;
        else if (slot < cap_tot) gc[slot - CAP_SMEM] = k;
    }
}

template<int BT, class Load, class F>
__device__ __forceinline__ void each4(int n4, Load load, F f) {
    constexpr int U = 4;
    for (int b = 0; b < n4; b += BT * U) {
        uint64_t k[U][4];
        #pragma unroll
        for (int u = 0; u < U; ++u) load(b + u * BT + (int)threadIdx.x, k[u]);
        #pragma unroll
        for (int u = 0; u < U; ++u)
            #pragma unroll
            for (int e = 0; e < 4; ++e) f(k[u][e]);
    }
}

template<int BT>
struct Row {
    SmemSel<BT>& m; const float* x; int len; bool v4; uint64_t* gc; int cap_tot;

    __device__ void clear(bool zero_cand) {
        for (int j = threadIdx.x; j < NB; j += BT) m.hist[j] = 0;
        if (threadIdx.x == 0 && zero_cand) m.s.ncand = 0;
        __syncthreads();
    }
    __device__ void choose() {
        constexpr int G = NB / BT;
        const int t = threadIdx.x;
        int c = 0;
        #pragma unroll
        for (int j = 0; j < G; ++j) c += m.hist[NB - 1 - t * G - j];
        int before, total;
        cub::BlockScan<int, BT, cub::BLOCK_SCAN_WARP_SCANS>(m.scan).ExclusiveSum(c, before, total);
        const int need = m.s.need;
        if (before < need && before + c >= need) {
            int acc = before;
            for (int j = 0; j < G; ++j) {
                const int b = NB - 1 - t * G - j, h = m.hist[b];
                if (acc + h >= need) { m.s.B = b; m.s.above = acc; m.s.hb = h; break; }
                acc += h;
            }
        }
        if (t == 0) m.s.total = total;
        __syncthreads();
    }
    __device__ void descend() {
        St& s = m.s;
        s.pp = s.p; s.lp = s.l;
        if (s.total < s.need) { s.fin = 1; return; }
        const int w = min(W, 64 - s.l);
        s.p = s.l ? (s.p << w) | (uint64_t)s.B : (uint64_t)s.B;
        s.l += w; s.need -= s.above; s.fin = (s.hb == s.need);
    }
    __device__ void pass(int src, bool emit_ok, bool collect, int samp_stride, int sb) {
        const uint64_t pp = m.s.pp, p = m.s.p;
        const int lp = m.s.lp, l = m.s.l, fin = m.s.fin;
        const int w = min(W, 64 - l), sh = 64 - l - w;
        const uint32_t dmask = (1u << w) - 1u;
        auto f = [&](uint64_t k) {
            const bool in = (k != 0ull) && (lp == 0 || (k >> (64 - lp)) == pp);
            bool em = false, hs = false;
            if (in) {
                const uint64_t u = l ? (k >> (64 - l)) : 0ull;
                if (fin) em = (l == lp) || (u >= p);
                else { em = (l != 0) && (u > p); hs = (l == 0) || (u == p); }
                em &= emit_ok;
            }
            append_out(em, k, &m.s.nout, m.out);
            if (__any_sync(0xffffffffu, hs)) {
                const unsigned d = hs ? (unsigned(k >> sh) & dmask) : (unsigned)NB;
                const unsigned peers = __match_any_sync(0xffffffffu, d);
                if (hs && (threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(m.hist + d, __popc(peers));
                if (collect) append_cand(hs, k, &m.s.ncand, m.cand, gc, cap_tot);
            }
        };
        if (src == 0) {
            each4<BT>((len + 3) / 4, [&](int i4, uint64_t* k) { load_row4(x, len, v4, 4 * i4, k); }, f);
        } else if (src == 1) {
            const int sb4 = sb >> 2;
            const int nblk = (len + sb - 1) / sb;
            const int ns = (nblk + samp_stride - 1) / samp_stride;
            each4<BT>(ns * sb4, [&](int i4, uint64_t* k) {
                const int i0 = (i4 / sb4) * sb * samp_stride + (i4 % sb4) * 4;
                load_row4(x, len, v4, i0, k);
            }, f);
        } else {
            const int n = m.s.ncand;
            each4<BT>((n + 3) / 4, [&](int i4, uint64_t* k) {
                const int b = 4 * i4;
                if (b + 3 < n && b + 3 < CAP_SMEM) {
                    const ulonglong2 v0 = *reinterpret_cast<const ulonglong2*>(m.cand + b);
                    const ulonglong2 v1 = *reinterpret_cast<const ulonglong2*>(m.cand + b + 2);
                    k[0] = v0.x; k[1] = v0.y; k[2] = v1.x; k[3] = v1.y;
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int idx = b + e;
                        if (idx >= n) k[e] = 0ull;
                        else if (idx < CAP_SMEM) k[e] = m.cand[idx];
                        else k[e] = gc[idx - CAP_SMEM];
                    }
                }
            }, f);
        }
    }
    __device__ void levels(int src) {
        for (;;) {
            const bool collect = (src == 0) && (m.s.l > 0);
            clear(collect);
            pass(src, true, collect, 1, 256);
            __syncthreads();
            if (m.s.fin) return;
            choose();
            const bool to_cand = collect && (m.s.total <= cap_tot);
            __syncthreads();
            if (threadIdx.x == 0) descend();
            __syncthreads();
            if (to_cand) src = 2;
        }
    }
};

template<int BT>
__global__ void __launch_bounds__(BT, 2) select_rows_v2(
    const float* __restrict__ scores, int64_t stride, int len,
    const int* __restrict__ positions, int pos_offset,
    int* __restrict__ indices, uint64_t* __restrict__ cand_g, int capg)
{
    constexpr int ITEMS = TOP / BT;
    using SortT = cub::BlockRadixSort<uint64_t, BT, ITEMS>;
    extern __shared__ __align__(16) unsigned char raw_smem[];
    Smem<BT>& smu = *reinterpret_cast<Smem<BT>*>(raw_smem);
    SmemSel<BT>& m = smu.sel;

    const int r = blockIdx.x, t = threadIdx.x;
    const int n = positions ? max(0, min(len, positions[pos_offset + r] + 1)) : len;
    const float* x = scores + (int64_t)r * stride;
    const bool v4 = (((uintptr_t)x & 15u) == 0u);
    uint64_t* gc = cand_g + (int64_t)r * capg;
    const int cap_tot = CAP_SMEM + capg;

    if (n <= TOP) {
        uint64_t keys[ITEMS];
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const int id = t * ITEMS + j;
            keys[j] = (id < n) ? make_key(__ldg(x + id), id) : 0ull;
        }
        SortT(smu.sort).SortDescending(keys);
        int* o = indices + (int64_t)r * TOP;
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const uint64_t k = keys[j];
            o[t * ITEMS + j] = k ? int(0xffffffffu - uint32_t(k)) : -1;
        }
        return;
    }

    Row<BT> rw{m, x, n, v4, gc, cap_tot};
    if (t == 0) { m.s = St{}; m.s.need = TOP; }
    __syncthreads();

    const int samp = (n >= 98304) ? 32 : (n >= 49152 ? 16 : (n >= 20480 ? 8 : 4));
    constexpr int SB = 256;
    const int mu = TOP / samp;
    const int k_samp = mu + ((samp == 32) ? 32 : (samp == 16 ? 48 : (samp == 8 ? 64 : 128)));

    if (t == 0) m.s.need = k_samp;
    __syncthreads();

    uint64_t bound = 1ull;
    rw.clear(false);
    rw.pass(1, false, false, samp, SB);
    __syncthreads();
    rw.choose();
    if (m.s.total >= k_samp) {
        const uint64_t B0 = m.s.B;
        const int need1 = k_samp - m.s.above;
        bound = B0 << (64 - W);
        if (m.s.hb != need1) {
            __syncthreads();
            if (t == 0) { m.s.p = B0; m.s.l = W; m.s.need = need1; }
            rw.clear(false);
            rw.pass(1, false, false, samp, SB);
            __syncthreads();
            rw.choose();
            bound = ((B0 << W) | (uint64_t)m.s.B) << (64 - 2 * W);
        }
    }
    __syncthreads();
    if (t == 0) { m.s = St{}; m.s.need = TOP; }
    __syncthreads();

    const float bound_f = (bound == 1ull) ? -3.4028234663852886e+38f : inv_ord((uint32_t)(bound >> 32));
    const int n4 = (n + 3) / 4;
    constexpr int UF = 4;
    for (int b = 0; b < n4; b += BT * UF) {
        float4 qv[UF];
        int i0s[UF];
        #pragma unroll
        for (int u = 0; u < UF; ++u) {
            const int i0 = 4 * (b + u * BT + t);
            i0s[u] = i0;
            if (v4 && i0 + 3 < n) {
                qv[u] = __ldg(reinterpret_cast<const float4*>(x + i0));
            } else {
                qv[u].x = (i0 + 0 < n) ? __ldg(x + i0 + 0) : -INFINITY;
                qv[u].y = (i0 + 1 < n) ? __ldg(x + i0 + 1) : -INFINITY;
                qv[u].z = (i0 + 2 < n) ? __ldg(x + i0 + 2) : -INFINITY;
                qv[u].w = (i0 + 3 < n) ? __ldg(x + i0 + 3) : -INFINITY;
            }
        }
        #pragma unroll
        for (int u = 0; u < UF; ++u) {
            const float4 q = qv[u];
            const bool p0 = (q.x >= bound_f) && (q.x < INFINITY);
            const bool p1 = (q.y >= bound_f) && (q.y < INFINITY);
            const bool p2 = (q.z >= bound_f) && (q.z < INFINITY);
            const bool p3 = (q.w >= bound_f) && (q.w < INFINITY);
            if (__any_sync(0xffffffffu, p0 | p1 | p2 | p3)) {
                const int i0 = i0s[u];
                if (__any_sync(0xffffffffu, p0)) append_cand(p0, p0 ? make_key(q.x, i0 + 0) : 0ull, &m.s.ncand, m.cand, gc, cap_tot);
                if (__any_sync(0xffffffffu, p1)) append_cand(p1, p1 ? make_key(q.y, i0 + 1) : 0ull, &m.s.ncand, m.cand, gc, cap_tot);
                if (__any_sync(0xffffffffu, p2)) append_cand(p2, p2 ? make_key(q.z, i0 + 2) : 0ull, &m.s.ncand, m.cand, gc, cap_tot);
                if (__any_sync(0xffffffffu, p3)) append_cand(p3, p3 ? make_key(q.w, i0 + 3) : 0ull, &m.s.ncand, m.cand, gc, cap_tot);
            }
        }
    }
    __syncthreads();

    if (m.s.ncand <= cap_tot && (m.s.ncand >= TOP || bound == 1ull)) {
        if (m.s.ncand <= TOP) {
            for (int j = t; j < m.s.ncand; j += BT) m.out[j] = (j < CAP_SMEM) ? m.cand[j] : gc[j - CAP_SMEM];
            if (t == 0) m.s.nout = m.s.ncand;
        } else {
            rw.levels(2);
        }
    } else {
        __syncthreads();
        if (t == 0) { m.s = St{}; m.s.need = TOP; }
        __syncthreads();
        rw.levels(0);
    }
    __syncthreads();
    const int nout = min(m.s.nout, TOP);

    uint64_t keys[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int idx = t * ITEMS + j;
        keys[j] = (idx < nout) ? m.out[idx] : 0ull;
    }
    __syncthreads();
    SortT(smu.sort).SortDescending(keys);

    int* o = indices + (int64_t)r * TOP;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const uint64_t k = keys[j];
        o[t * ITEMS + j] = k ? int(0xffffffffu - uint32_t(k)) : -1;
    }
}

template<int BT>
static inline cudaError_t select(
    const float* scores, int64_t stride, int len,
    const int* positions, int pos_offset, int rows,
    int* indices, uint64_t* cand_g, int capg, cudaStream_t st)
{
    if (rows <= 0) return cudaSuccess;
    static const bool once =
        cudaFuncSetAttribute(select_rows_v2<BT>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem<BT>)) == cudaSuccess;
    (void)once;
    select_rows_v2<BT><<<rows, BT, sizeof(Smem<BT>), st>>>(scores, stride, len, positions, pos_offset, indices, cand_g, capg);
    return cudaGetLastError();
}

} // namespace selopt2

static inline cudaError_t select_prefill(const G6Attn& a, ga::Work& w, int start, int n, cudaStream_t st) {
    uint64_t* cand_g = w.so ? reinterpret_cast<uint64_t*>(w.so) : reinterpret_cast<uint64_t*>(w.ii);
    const int capg = w.so ? std::min(selx::CAPG, a.context) : std::min(selx::CAPG, a.context / 2);
    return selopt2::select<256>(w.si, a.context, a.context, a.positions, start, n,
                                a.indices + (int64_t)start * ga::TOP, cand_g, capg, st);
}

#if G6_KV4
// ============================================================================
// 4. Vectorized INT4 KV -> FP16 [576] table prepass (matches gqkv4::prepass)
// ============================================================================
template<bool FOLD = false>
__global__ void __launch_bounds__(256) prepass_opt(
    const uint8_t* __restrict__ records, half* __restrict__ table, uint8_t* __restrict__ bad, int tokens)
{
    __shared__ uint8_t sm_q[8][256];
    __shared__ half sm_sc[8][16];
    __shared__ half sm_out[8][512];

    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int key = blockIdx.x * 8 + w;
    if (key >= tokens) return;
    const uint8_t* p = records + (int64_t)key * g6kv::ROW;
    half* row = table + (int64_t)key * 576;

    reinterpret_cast<uint2*>(sm_q[w])[lane] = reinterpret_cast<const uint2*>(p)[lane];
    if (lane < 16) sm_sc[w][lane] = reinterpret_cast<const half*>(p + g6kv::SCALE_OFF)[lane];
    __syncwarp();

    float x[16];
    const int byte_off = lane >> 1;
    const int nib_sh = (lane & 1) * 4;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const uint8_t b = sm_q[w][byte_off + 16 * k];
        x[k] = (float)(((b >> nib_sh) & 15) - 8) * __half2float(sm_sc[w][k]);
    }
    if constexpr (!FOLD) ga::fwht16(x, lane);

    bool nf = false;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const half h = __float2half_rn(FOLD ? x[k] : x[k] * ga::HSCALE);
        nf |= !isfinite(__half2float(h));
        sm_out[w][lane + 32 * k] = h;
    }
    __syncwarp();

    reinterpret_cast<uint4*>(row)[lane]      = reinterpret_cast<const uint4*>(sm_out[w])[lane];
    reinterpret_cast<uint4*>(row)[lane + 32] = reinterpret_cast<const uint4*>(sm_out[w])[lane + 32];

    const __nv_bfloat162 b2 = reinterpret_cast<const __nv_bfloat162*>(p + g6kv::ROPE_OFF)[lane];
    const half h0 = __float2half_rn(__bfloat162float(b2.x));
    const half h1 = __float2half_rn(__bfloat162float(b2.y));
    nf |= !isfinite(__half2float(h0)) | !isfinite(__half2float(h1));
    reinterpret_cast<__half2*>(row + 512)[lane] = __halves2half2(h0, h1);

    nf = __any_sync(0xffffffffu, nf);
    if (lane == 0) bad[key] = (uint8_t)nf;
}
#endif

} // namespace pf_attn_opt
