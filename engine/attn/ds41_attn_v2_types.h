// V2-only attention staging. InputBuf reuses gr8 PR2 (0b35ab7) OwnBuf layout/allocator,
// with one ratio-1 compressor latent. No stock/decode path uses these buffers.
#pragma once
#include "ds41_attn_types.h"
namespace ds41a { namespace v2 {
// mode: 0 = qa/kv; 1 = + ratio-1 compressor latent; 2 = + ratio-2 compressor pair.
__host__ __device__ constexpr int input_width(int mode) { return QA + HD + mode * HD; }
__host__ __device__ constexpr int input_stride(int mode) { return (input_width(mode) * 33 / 32 + 63) & ~63; }
__host__ __device__ constexpr int output_width(int heads) { return heads / HPG * OLORA; }
__host__ __device__ constexpr int output_stride(int heads) { return output_width(heads) * 33 / 32; }
struct InputBuf
{
    bf16 *qa = nullptr, *kv0 = nullptr;
    float *ckv = nullptr, *csc = nullptr;
    bf16* ckv1 = nullptr; // ratio-1 compressor, layer 20
    uint8_t *yq = nullptr, *ys = nullptr;
    uint32_t* xf = nullptr;
    static size_t al(size_t b) { return (b + 255) & ~(size_t)255; }
    static size_t bytes(int rows, int q)
    {
        return al((size_t)rows * QA * 2) + al((size_t)rows * HD * 2) + 2 * al((size_t)rows * HD * 4) + al((size_t)rows * HD * 2) + al((size_t)q * HID) +
               al((size_t)q * HID / 32) + al((size_t)HID * 16);
    }
    void carve(void* base, int rows, int q)
    {
        uint8_t* p = (uint8_t*)base;
        auto take = [&](size_t b) { uint8_t* r = p; p += al(b); return r; };
        qa = (bf16*)take((size_t)rows * QA * 2);
        kv0 = (bf16*)take((size_t)rows * HD * 2);
        ckv = (float*)take((size_t)rows * HD * 4);
        csc = (float*)take((size_t)rows * HD * 4);
        ckv1 = (bf16*)take((size_t)rows * HD * 2);
        yq = take((size_t)q * HID);
        ys = take((size_t)q * HID / 32);
        xf = (uint32_t*)take((size_t)HID * 16);
    }
};

}} // namespace ds41a::v2
