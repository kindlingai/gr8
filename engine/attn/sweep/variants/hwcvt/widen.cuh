#pragma once
// Our hardware-conversion experiment. The existing BF16 scale multiply is kept.
namespace hwcvt {
__device__ __forceinline__ uint32_t widen_pair(uint32_t codes)
{
    uint32_t h;
    asm volatile("{ .reg .b8 a,b,c,d; mov.b32 {a,b,c,d}, %1; "
                 "cvt.rn.f16x2.e2m1x2 %0, a; }" : "=r"(h) : "r"(codes));
    const __half2 hh = *reinterpret_cast<const __half2*>(&h);
    const float2 f = __half22float2(hh);
    const __nv_bfloat162 v = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ void dq4(uint32_t c, __nv_bfloat162 s2,
                                  uint32_t& o0, uint32_t& o1)
{
    const __nv_bfloat162 a = __hmul2(ds41a::u32_bf2(widen_pair(c)), s2);
    const __nv_bfloat162 b = __hmul2(ds41a::u32_bf2(widen_pair(c >> 8)), s2);
    o0 = *reinterpret_cast<const uint32_t*>(&a);
    o1 = *reinterpret_cast<const uint32_t*>(&b);
}
}
