// G6_DRAFT_HEAD_I4: the MTP draft's lm_head as INT4 g32 GEMV records (densev.cuh Int4v, 288 B per 16 rows x 32 inputs),
// made on the GPU at boot from this rank's MXFP8 records of the same rows (G.lm_head.mx.gv, densev.cuh Mxfp8, 528 B).
// The target's head keeps the MXFP8 records; only the draft reads these. 3456 B per vocab row instead of 6336: the draft
// head reads 45% fewer bytes, on the same vocab slice, with the same head kernels (full-vocab argmax and log-probability).
//
// Per row and group of 32 inputs (engine/loader-rs records::int4_quantize on the MXFP8 values):
//   v = e4m3(code) * 2^(e8m0 - 127)   (exact in fp32)
//   s = fp16(amax(|v|) / 7)            round to nearest even
//   q = clamp(rint_even(v / s), -8, 7) + 8, q = 8 where s == 0
// Record layouts (engine/ref/g6ref.py mxfp8_record / rt_record), r = row & 7, hi = row >> 3, lane (r, c):
//   MXFP8  tile h (k 16h..16h+15) at 256 h, lane at 8 lane: {(r,2c) (r,2c+1) (r+8,2c) (r+8,2c+1) (r,2c+8) (r,2c+9)
//          (r+8,2c+8) (r+8,2c+9)}; E8M0 scale of row r + 8 hi at 512 + 2 r + hi
//   INT4   tile h at 128 h, lane at 4 lane: bytes {row r, row r+8} of k (2c, 2c+1), then of k (2c+8, 2c+9); the lower
//          k in the low nibble; fp16 scale of row r + 8 hi at 256 + 2 (2 r + hi)
// One thread per (16-row tile, group, row): it writes only its own row's bytes, so the conversion is deterministic.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace head_i4
{
constexpr int MXGR = 528, I4GR = 288;

__device__ __forceinline__ float e4m3_val(uint32_t b)
{
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e ? ldexpf((float)(8 + m), e - 10) : ldexpf((float)m, -9);
    return (b & 0x80u) ? -v : v;
}
__device__ __forceinline__ float e8m0_val(uint32_t code) { return __uint_as_float(code ? code << 23 : 0x00400000u); }

// mx [tiles][ng][528] -> i4 [tiles][ng][288]; grid ceil(tiles * ng * 16 / 256), block 256
__global__ void __launch_bounds__(256) k_mx_to_i4(const uint8_t* __restrict__ mx, uint8_t* __restrict__ i4, int tiles, int ng)
{
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)tiles * ng * 16) return;
    const int row = (int)(id & 15);
    const long long grp = id >> 4;                // tile * ng + group
    const uint8_t* src = mx + grp * MXGR;
    uint8_t* dst = i4 + grp * I4GR;
    const int r = row & 7, hi = row >> 3;
    const float sc = e8m0_val(src[512 + 2 * r + hi]);
    float v[32];
    float amax = 0.f;
    #pragma unroll
    for (int kk = 0; kk < 32; ++kk)
    {
        const int h = kk >> 4, k = kk & 15, c = (k & 7) >> 1;
        v[kk] = e4m3_val(src[h * 256 + (r * 4 + c) * 8 + (k >> 3) * 4 + hi * 2 + (k & 1)]) * sc;
        amax = fmaxf(amax, fabsf(v[kk]));
    }
    const __half s16 = __float2half_rn(__fdiv_rn(amax, 7.f));
    const float s = __half2float(s16);
    #pragma unroll
    for (int kk = 0; kk < 32; kk += 2)
    {
        const int h = kk >> 4, k = kk & 15, c = (k & 7) >> 1;
        uint32_t q0 = 8, q1 = 8;
        if (s != 0.f)
        {
            q0 = (uint32_t)(min(max(__float2int_rn(__fdiv_rn(v[kk], s)), -8), 7) + 8);
            q1 = (uint32_t)(min(max(__float2int_rn(__fdiv_rn(v[kk + 1], s)), -8), 7) + 8);
        }
        dst[h * 128 + (r * 4 + c) * 4 + (k >> 3) * 2 + hi] = (uint8_t)(q0 | (q1 << 4));
    }
    reinterpret_cast<__half*>(dst + 256)[2 * r + hi] = s16;
}

inline size_t bytes(long long tiles, int ng) { return (size_t)tiles * ng * I4GR; }
inline void launch(const uint8_t* mx, uint8_t* i4, int tiles, int ng, cudaStream_t st)
{
    const long long n = (long long)tiles * ng * 16;
    k_mx_to_i4<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(mx, i4, tiles, ng);
}
}   // namespace head_i4
