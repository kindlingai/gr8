// g6 M1 kernels: shared PTX helpers (GB10, sm_121a). Our own code.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_fp16.h>
#include "../g6nr.h"
#include "../g6mx.h"

typedef uint32_t u32;
typedef uint64_t u64;

constexpr int HID = 6144;            // hidden size
constexpr int KSH = HID / 16;        // 384 k16 steps over hidden
constexpr float SQ128 = 0.088388347648f;

// streaming loads: weights are read once, keep them out of L1
__device__ __forceinline__ u32 ld_nc(const void* p)
{
    u32 v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ u64 ld_nc64(const void* p)
{
    u64 v;
    asm volatile("ld.global.nc.L1::no_allocate.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}

// L2 warm-up: one prefetch.global.L2 per 64 B line of [p, p + 64 * nlines). grid = ceil(nlines / 2048), block 256.
// The old body (ld_nc, result unused) compiled to LDC + EXIT,
// 0 loads, so G6_WARM was a no-op. prefetch.global.L2 is asm volatile with a memory side effect, so it is kept from DCE
// (SASS: 8x CCTL.E.PF2); it moves whole 64 B lines (both sectors) into the GPU L2 and writes no register or memory.
// Changes no result. Measured: -9.6..-9.9 us/layer router window = 0.73-0.75 ms/step @M12 true-cold, ONLY when hidden.
// Use: issue the warm about ONE LAYER AHEAD of the kernel that reads the bytes, so it completes before that
// window. Not in the same window (in-stream unhidden: +5 us/layer hot, fresh 52.7 vs 45 us = loss) and not merely
// concurrent on a side stream with the collective wait (only -3.4 us fresh, +2.7 us every hot step = net ~0).
__global__ void __launch_bounds__(256) k_warm(const uint8_t* p, long long nlines)
{
    const long long t = blockIdx.x * 256ll + threadIdx.x;
    #pragma unroll
    for (int u = 0; u < 8; ++u)
    {
        const long long i = t * 8 + u;
        if (i < nlines) asm volatile("prefetch.global.L2 [%0];" :: "l"(p + i * 64) : "memory");
    }
}
inline void launch_warm(const void* p, size_t bytes, cudaStream_t st)
{
    if (!p || !bytes) return;
    const long long nl = static_cast<long long>((bytes + 63) / 64);
    k_warm<<<static_cast<unsigned>((nl + 2047) / 2048), 256, 0, st>>>(static_cast<const uint8_t*>(p) - (reinterpret_cast<size_t>(p) & 3), nl);
}
__device__ __forceinline__ u32 ld_nc16(const void* p)
{
    u32 v;
    asm volatile("{ .reg .u16 t; ld.global.nc.L1::no_allocate.u16 t, [%1]; cvt.u32.u16 %0, t; }" : "=r"(v) : "l"(p));
    return v;
}

// mma m16n8k16, fp16 operands, fp32 accumulate. A: 16 rows x 16 k, B: 16 k x 8 cols.
//   A regs: a0 (row l/4,   k 2c..2c+1), a1 (row l/4+8, same k), a2 (row l/4, k+8), a3 (row l/4+8, k+8)
//   B regs: b0 (k 2c..2c+1, col l/4), b1 (k+8)
//   C regs: c0,c1 (row l/4, col 2c, 2c+1), c2,c3 (row l/4+8)          with l = lane, c = l%4
__device__ __forceinline__ void mma16816(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ u32 e2m1x2(u32 byte)   // two FP4 E2M1 nibbles -> half2 (low nibble = low half)
{
    u32 r;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "r"(byte));
    return r;
}
__device__ __forceinline__ u32 e4m3x2(u32 two)    // two FP8 E4M3 bytes -> half2
{
    u32 r;
    asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(two));
    return r;
}

__device__ __forceinline__ __half2 as_h2(u32 v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ u32 as_u32(__half2 h) { return *reinterpret_cast<u32*>(&h); }
__device__ __forceinline__ u32 pack_h2(float a, float b) { return as_u32(__floats2half2_rn(a, b)); }
__device__ __forceinline__ uint16_t f2bf(float f)   // fp32 -> bf16 bits, round to nearest even (finite input)
{
    u32 u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { return static_cast<u32>(f2bf(a)) | (static_cast<u32>(f2bf(b)) << 16); }
__device__ __forceinline__ u32 pack_o2(float a, float b, int bf16) { return bf16 ? pack_bf2(a, b) : pack_h2(a, b); }
__device__ __forceinline__ void put16(__half* y, float v, int bf16)   // one 16-bit output cell, fp16 or BF16 bits
{
    if (bf16) *reinterpret_cast<uint16_t*>(y) = f2bf(v);
    else *y = __float2half_rn(v);
}
__device__ __forceinline__ float rnd_h(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }

// 128-point Hadamard (Sylvester order), one row held by a warp as 4 consecutive floats per lane,
// scaled by 1/sqrt(128).
__device__ __forceinline__ void had128(float (&v)[4], int lane)
{
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1)
    {
        const bool hi = (lane & m) != 0;
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float p = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = hi ? (p - v[e]) : (v[e] + p);
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) v[e] *= SQ128;
}

// A warp holds one row of a 128 block as 4 consecutive values per lane (index lane*4+e).
// Store it into an mma fragment table laid out [k16 step][lane'][2 regs] (u32 = half2), where the row
// lands in fragment slot `row` (0..7): this is the A-operand layout (rows) and also the B-operand
// layout (cols) of mma16816.  ks_base = k16 step of element 0.
__device__ __forceinline__ void store_frag_row(u32* tab, int ks_base, int row, int lane, const float (&v)[4])
{
    const int ks = ks_base + (lane >> 2);
    const int q = lane & 3;               // elements q*4 .. q*4+3 of the 16-wide step
    const int reg = q >> 1;               // k 0-7 -> reg 0, k 8-15 -> reg 1
    const int c = (q & 1) * 2;            // pair index inside the 8
    u32* p = tab + ((ks * 32 + row * 4 + c) * 2 + reg);
    p[0] = pack_h2(v[0], v[1]);
    p[2] = pack_h2(v[2], v[3]);           // next lane' (c+1), same reg
}
