// The device adapters that put the DS41 shared expert (prefill, rows >= RO_FROM) on the attention block's tensor-core MXFP8 GEMM (k_gemm_mx8t,
// ds41_gemm_mx_tc): adapters A1 and A3 below; no GEMM here, the chain reuses ds41_gemm_mx_tc.
// Included by ds41_attn.cu only (inside this TU: act_quant_lane is here, and the C entries in ds41_attn.cu are the rank TU's way in).
//   A1  k_mx_unrecord: the inverse of k_kda_mx_record (engine/kda/kda.cuh, = loader-rs records::mxfp8_record), the same index map with source and
//       target swapped: one 528 B record [16 rows][32 k] -> plain codes rows [16][32] (row stride ldc) + E8M0 bytes [16] (row stride lds).
//       The caller zeroes the target first when it pads (fc2: K 576 -> 640; codes 0 add exactly 0 whatever the scale byte is).
//   A3  k_sh_swiglu_q: Y13 bf16 [M][2I] (gate cols 0..I-1, up cols I..2I-1) -> act = swiglu_lim(gate, up, L) (the function every fc1 of the
//       GL path uses, ../kernels/swiglu_lim.cuh), quantized as k_act_quant does (act_quant_lane): codes u8 [M][KP] + E8M0 bytes [M][KP/32];
//       blocks b >= I/32 (the pad up to KP) are code 0, byte 0. One warp per 32-block.
#pragma once
#include "../kernels/swiglu_lim.cuh"
#include "ds41_sh_tc_map.h"

namespace ds41a
{
// grid = T * ng records (ng = K / 32), block 256. rec [T][ng][528] -> codes [16T][ldc] (cols 0..K-1) + scales [16T][lds] (cols 0..ng-1)
__global__ void __launch_bounds__(256) k_mx_unrecord(const uint8_t* __restrict__ rec_all, int ng, uint8_t* __restrict__ codes, int ldc,
                                                     uint8_t* __restrict__ scales, int lds)
{
    const int t = blockIdx.x / ng, g = blockIdx.x % ng, i = threadIdx.x;
    const uint8_t* rec = rec_all + static_cast<size_t>(blockIdx.x) * MX_REC_B;
    int row, col;
    mx_rec_rc(i, row, col);
    for (int h = 0; h < 2; ++h)
        codes[static_cast<size_t>(16 * t + row) * ldc + 32 * g + 16 * h + col] = rec[h * 256 + i];
    if (i < 8)
    {
        scales[static_cast<size_t>(16 * t + i) * lds + g] = rec[512 + 2 * i];
        scales[static_cast<size_t>(16 * t + i + 8) * lds + g] = rec[512 + 2 * i + 1];
    }
}
// grid = ceil(M * KP / 32 / 8), block 256 (8 warps, a warp per (row, 32-block)).
__global__ void k_sh_swiglu_q(const bf16* __restrict__ y13, int ldy, int I, float lim, uint8_t* __restrict__ codes, int ldc,
                              uint8_t* __restrict__ sc, int lds, int M, int KP)
{
    const int lane = threadIdx.x & 31;
    const int kb = KP >> 5;
    const long w = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= (long)M * kb) return;
    const int r = (int)(w / kb), b = (int)(w % kb);
    if (b * 32 >= I)                                   // the pad blocks (K 576 -> 640): code 0, byte 0
    {
        codes[(size_t)r * ldc + b * 32 + lane] = 0;
        if (lane == 0) sc[(size_t)r * lds + b] = 0;
        return;
    }
    const bf16* yr = y13 + (size_t)r * ldy + b * 32 + lane;
    const float v = mx4::swiglu_lim(__bfloat162float(yr[0]), __bfloat162float(yr[I]), lim);
    uint8_t c, e;
    act_quant_lane(v, c, e);
    codes[(size_t)r * ldc + b * 32 + lane] = c;
    if (lane == 0) sc[(size_t)r * lds + b] = e;
}
}   // namespace ds41a
