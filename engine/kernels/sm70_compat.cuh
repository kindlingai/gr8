// SM70 (NVIDIA Volta / Tesla V100) compatibility primitives for gr8.
// On __CUDA_ARCH__ >= 800/890/1200, every helper compiles to the original native PTX instructions
// with zero overhead. On __CUDA_ARCH__ < 800 (sm_70):
//   - Streaming / L2 cache-hint loads & stores degrade cleanly to __ldg() / normal accesses.
//   - FP8 E4M3 / FP4 E2M1 conversions use bit-exact SM70 software helpers (<cuda_fp8.h> / exact bit-ops).
//   - cp.async degrades to synchronous shared-memory copies.
//   - ldmatrix.sync.aligned.m8n8.x{2,4}[.trans] is emulated via warp shuffles + shared loads.
//   - mma.sync.aligned.m16n8k16 (FP16 and BF16) executes on Volta's real hardware Tensor Cores via
//     two mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 instructions + 8-shuffle butterfly reduction
//     (zero shared memory, 100% compatible with callers' m16n8k16 A/B/C register layouts).
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#endif

namespace gr8_sm70
{
#ifdef __CUDACC__

// ---------------------------------------------------------------------------------------------- streaming loads/stores
__device__ __forceinline__ uint32_t ld_nc_u32(const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
#else
    return __ldg(static_cast<const uint32_t*>(p));
#endif
}

__device__ __forceinline__ uint64_t ld_nc_u64(const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint64_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
#else
    return __ldg(static_cast<const uint64_t*>(p));
#endif
}

__device__ __forceinline__ uint32_t ld_nc_u16_to_u32(const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t v;
    asm volatile("{ .reg .u16 t; ld.global.nc.L1::no_allocate.u16 t, [%1]; cvt.u32.u16 %0, t; }" : "=r"(v) : "l"(p));
    return v;
#else
    return static_cast<uint32_t>(__ldg(static_cast<const uint16_t*>(p)));
#endif
}

__device__ __forceinline__ void ld_nc_v2_u32(const void* p, uint32_t* v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v[0]), "=r"(v[1]) : "l"(p));
#else
    const uint2 t = __ldg(static_cast<const uint2*>(p));
    v[0] = t.x;
    v[1] = t.y;
#endif
}

__device__ __forceinline__ void ld_nc_v4_u32(const void* p, uint32_t* v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(p));
#else
    const uint4 t = __ldg(static_cast<const uint4*>(p));
    v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
#endif
}

__device__ __forceinline__ uint4 ld_nc_uint4(const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
#else
    return __ldg(static_cast<const uint4*>(p));
#endif
}

__device__ __forceinline__ uint4 ld_nc_uint4_l2_256b(const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
#else
    return __ldg(static_cast<const uint4*>(p));
#endif
}

__device__ __forceinline__ uint4 ld_nc_uint4_ef(const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint4 v;
    asm volatile(
        "{\n\t.reg .b64 pol;\n\t"
        "createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
        "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], pol;\n\t}"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
#else
    return __ldg(static_cast<const uint4*>(p));
#endif
}

__device__ __forceinline__ uint64_t ld_nc_u64_ef(const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint64_t v;
    asm volatile(
        "{\n\t.reg .b64 pol;\n\t"
        "createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
        "ld.global.nc.L1::no_allocate.L2::cache_hint.u64 %0, [%1], pol;\n\t}"
        : "=l"(v) : "l"(p));
    return v;
#else
    return __ldg(static_cast<const uint64_t*>(p));
#endif
}

__device__ __forceinline__ void st_uint4_el(void* p, uint4 v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "{\n\t.reg .b64 pol;\n\t"
        "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
        "st.global.L2::cache_hint.v4.u32 [%0], {%1,%2,%3,%4}, pol;\n\t}"
        :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
#else
    *static_cast<uint4*>(p) = v;
#endif
}

__device__ __forceinline__ void st_uint4_ef(void* p, uint4 v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "{\n\t.reg .b64 pol;\n\t"
        "createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
        "st.global.L2::cache_hint.v4.u32 [%0], {%1,%2,%3,%4}, pol;\n\t}"
        :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
#else
    *static_cast<uint4*>(p) = v;
#endif
}

__device__ __forceinline__ void st_u32_ef(void* p, uint32_t v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "{\n\t.reg .b64 pol;\n\t"
        "createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n\t"
        "st.global.L2::cache_hint.u32 [%0], %1, pol;\n\t}"
        :: "l"(p), "r"(v) : "memory");
#else
    *static_cast<uint32_t*>(p) = v;
#endif
}

__device__ __forceinline__ void discard_l2_128(void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory");
#else
    (void)p;
#endif
}

__device__ __forceinline__ void prefetch_l2_bulk(const void* p, uint32_t bytes)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
#else
    (void)bytes;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p) : "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ uint32_t pack_bf16x2_f32(float lo, float hi)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#else
    auto f2b = [](float f) -> uint32_t {
        uint32_t u = __float_as_uint(f);
        if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
        return u >> 16;
    };
    return f2b(lo) | (f2b(hi) << 16);
#endif
}

__device__ __forceinline__ void cp_async_cg_16_addr(uint32_t saddr, const void* gmem, int bytes)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(saddr), "l"(gmem), "r"(bytes) : "memory");
#else
    const uint4 v = (bytes >= 16) ? *static_cast<const uint4*>(gmem) : make_uint4(0u, 0u, 0u, 0u);
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(saddr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
#endif
}

__device__ __forceinline__ void ldmatrix_x4_addr(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t saddr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
#else
    const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const uint32_t off = static_cast<uint32_t>(c * 4);
    const uint32_t a0 = __shfl_sync(0xffffffffu, saddr, g) + off;
    const uint32_t a1 = __shfl_sync(0xffffffffu, saddr, g + 8) + off;
    const uint32_t a2 = __shfl_sync(0xffffffffu, saddr, g + 16) + off;
    const uint32_t a3 = __shfl_sync(0xffffffffu, saddr, g + 24) + off;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r0) : "r"(a0));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r1) : "r"(a1));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r2) : "r"(a2));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r3) : "r"(a3));
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans_addr(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t saddr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
#else
    const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const uint32_t off = static_cast<uint32_t>(g * 2);
    auto rd = [&](int base) -> uint32_t {
        const uint32_t p0 = __shfl_sync(0xffffffffu, saddr, base + 2 * c) + off;
        const uint32_t p1 = __shfl_sync(0xffffffffu, saddr, base + 2 * c + 1) + off;
        uint16_t lo, hi;
        asm volatile("ld.shared.u16 %0, [%1];" : "=h"(lo) : "r"(p0));
        asm volatile("ld.shared.u16 %0, [%1];" : "=h"(hi) : "r"(p1));
        return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    };
    r0 = rd(0); r1 = rd(8); r2 = rd(16); r3 = rd(24);
#endif
}

// ---------------------------------------------------------------------------------------------- FP16 vector helpers (SM80+ intrinsics)
__device__ __forceinline__ __half2 habs2(__half2 a)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __habs2(a);
#else
    uint32_t u = *reinterpret_cast<const uint32_t*>(&a);
    u &= 0x7fff7fffu;
    return *reinterpret_cast<const __half2*>(&u);
#endif
}

__device__ __forceinline__ __half2 hmax2(__half2 a, __half2 b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hmax2(a, b);
#else
    return __floats2half2_rn(fmaxf(__low2float(a), __low2float(b)), fmaxf(__high2float(a), __high2float(b)));
#endif
}

// ---------------------------------------------------------------------------------------------- FP8 / FP4 conversions
// Bit-exact E2M1 nibble -> FP16 bits (0..7 -> {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, bit 3 = sign)
__device__ __forceinline__ uint32_t e2m1_nib_to_f16(uint32_t nib)
{
    const uint32_t m = nib & 7u;
    const uint32_t mag = (m == 0u) ? 0u : ((m == 1u) ? 0x3800u : (0x3800u + (m << 9)));
    return mag | ((nib & 8u) << 12);
}

__device__ __forceinline__ uint32_t e2m1x2_to_f16x2(uint32_t byte)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t r;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "r"(byte));
    return r;
#else
    return e2m1_nib_to_f16(byte & 0xfu) | (e2m1_nib_to_f16((byte >> 4) & 0xfu) << 16);
#endif
}

// Bit-exact, loop-free E4M3 byte -> FP16 bits matching __nv_cvt_fp8_to_halfraw(..., __NV_E4M3) for all 256 bytes.
__device__ __forceinline__ uint32_t e4m3_byte_to_f16(uint32_t b)
{
    const uint32_t m = b & 0x7fu;
    const uint32_t s = (b & 0x80u) << 8;
    if (m == 0x7fu) return 0x7fffu;
    if (m >= 8u) return s | ((m << 7) + 0x2000u);
    const uint16_t sub = __half_as_ushort(
        __hsub(__ushort_as_half(static_cast<uint16_t>(0x2400u | (m << 7))), __ushort_as_half(0x2400u)));
    return s | static_cast<uint32_t>(sub);
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint32_t r;
    asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(two));
    return r;
#else
    return e4m3_byte_to_f16(two & 0xffu) | (e4m3_byte_to_f16((two >> 8) & 0xffu) << 16);
#endif
}

// Fast, loop-free, branch-free, bit-exact FP16x2 -> E4M3x2 (satfinite, RN-even) on SM70.
// Matches PTX cvt.rn.satfinite.e4m3x2.f16x2 for all 65,536 half values (verified exhaustively in sm70_mma_test).
__device__ __forceinline__ uint16_t f16x2_to_e4m3x2_sm70(uint32_t h2)
{
    const uint32_t m2 = h2 & 0x7fff7fffu;
    const uint32_t kTwo = 0x40004000u;
    const __half2 two = *reinterpret_cast<const __half2*>(&kTwo);
    const __half2 sum2 = __hadd2(*reinterpret_cast<const __half2*>(&m2), two);
    const uint32_t su2 = *reinterpret_cast<const uint32_t*>(&sum2);
    // Normal path: RN-even drop of low 7 mantissa bits (m < 0x5f00 -> zero cross-lane carry)
    const uint32_t norm2 = (m2 + 0x003f003fu + ((m2 >> 7) & 0x00010001u)) & 0x7f807f80u;
    auto cvt1 = [](uint32_t s, uint32_t m, uint32_t su, uint32_t nr) -> uint32_t {
        if (m > 0x7c00u) return 0x7fu;
        const uint32_t e = (m < 0x2400u) ? (su - 0x4000u)
                         : (m < 0x5f00u) ? ((nr - 0x2000u) >> 7)
                         : 0x7eu;
        return (s >> 8) | e;
    };
    const uint32_t lo = cvt1(h2 & 0x8000u, m2 & 0xffffu, su2 & 0xffffu, norm2 & 0xffffu);
    const uint32_t hi = cvt1((h2 >> 16) & 0x8000u, m2 >> 16, su2 >> 16, norm2 >> 16);
    return static_cast<uint16_t>(lo | (hi << 8));
}

__device__ __forceinline__ uint16_t f16x2_to_e4m3x2(uint32_t h2)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint16_t e;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e) : "r"(h2));
    return e;
#else
    return f16x2_to_e4m3x2_sm70(h2);
#endif
}

static __device__ __noinline__ uint32_t f32x2_to_e4m3x2_sm70(float a, float b)
{
    return static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint32_t f32x2_to_e4m3x2(float a, float b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint32_t r;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; cvt.u32.u16 %0, t; }" : "=r"(r) : "f"(a), "f"(b));
    return r;
#else
    return f32x2_to_e4m3x2_sm70(a, b);
#endif
}

// Fast, loop-free, branch-free, bit-exact round-to-E4M3-and-back for packed FP16x2 on SM70.
// Matches PTX cvt.rn.satfinite.e4m3x2.f16x2 + cvt.rn.f16x2.e4m3x2 for all 65,536 half values.
__device__ __forceinline__ uint32_t round_f16x2_e4m3_sm70(uint32_t h2)
{
    const uint32_t m2 = h2 & 0x7fff7fffu;
    const uint32_t kTwo = 0x40004000u;
    const __half2 two = *reinterpret_cast<const __half2*>(&kTwo);
    const __half2 subh = __hsub2(__hadd2(*reinterpret_cast<const __half2*>(&m2), two), two);
    const uint32_t sub2 = *reinterpret_cast<const uint32_t*>(&subh);
    const uint32_t norm2 = (m2 + 0x003f003fu + ((m2 >> 7) & 0x00010001u)) & 0x7f807f80u;
    auto rnd1 = [](uint32_t s, uint32_t m, uint32_t sub, uint32_t nr) -> uint32_t {
        if (m > 0x7c00u) return 0x7fffu;
        const uint32_t r = (m < 0x2400u) ? sub : ((m < 0x5f00u) ? nr : 0x5f00u);
        return s | r;
    };
    const uint32_t lo = rnd1(h2 & 0x8000u, m2 & 0xffffu, sub2 & 0xffffu, norm2 & 0xffffu);
    const uint32_t hi = rnd1((h2 >> 16) & 0x8000u, m2 >> 16, sub2 >> 16, norm2 >> 16);
    return lo | (hi << 16);
}

// Specialized tr3 codebook variant where |x| < 448.0 is mathematically guaranteed (MCG/MUL1/PMCGX <= 4.0).
__device__ __forceinline__ uint32_t round_f16x2_e4m3_bounded(uint32_t h2)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(h2) : "r"(h2));
    return h2;
#else
    const uint32_t s2 = h2 & 0x80008000u;
    const uint32_t m2 = h2 & 0x7fff7fffu;
    const uint32_t kTwo = 0x40004000u;
    const __half2 two = *reinterpret_cast<const __half2*>(&kTwo);
    const __half2 subh = __hsub2(__hadd2(*reinterpret_cast<const __half2*>(&m2), two), two);
    const uint32_t sub2 = *reinterpret_cast<const uint32_t*>(&subh);
    const uint32_t norm2 = (m2 + 0x003f003fu + ((m2 >> 7) & 0x00010001u)) & 0x7f807f80u;
    const uint32_t lo = ((m2 & 0xffffu) < 0x2400u) ? (sub2 & 0xffffu) : (norm2 & 0xffffu);
    const uint32_t hi = ((m2 >> 16) < 0x2400u) ? (sub2 >> 16) : (norm2 >> 16);
    return s2 | lo | (hi << 16);
#endif
}

__device__ __forceinline__ uint32_t round_f16x2_e4m3(uint32_t h2)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(h2) : "r"(h2));
    return h2;
#else
    return round_f16x2_e4m3_sm70(h2);
#endif
}

// ---------------------------------------------------------------------------------------------- cp.async & ldmatrix
__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* gmem)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
#else
    *static_cast<uint4*>(smem) = *static_cast<const uint4*>(gmem);
#endif
}

__device__ __forceinline__ void cp_async_cg_16_pred(void* smem, const void* gmem, int bytes)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(bytes) : "memory");
#else
    if (bytes >= 16) *static_cast<uint4*>(smem) = *static_cast<const uint4*>(gmem);
    else *static_cast<uint4*>(smem) = make_uint4(0u, 0u, 0u, 0u);
#endif
}

__device__ __forceinline__ void cp_async_ca_4(void* smem, const void* gmem)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem));
#else
    *static_cast<uint32_t*>(smem) = *static_cast<const uint32_t*>(gmem);
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s) : "memory");
#else
    const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const uintptr_t p = reinterpret_cast<uintptr_t>(smem_ptr);
    r0 = reinterpret_cast<const uint32_t*>(__shfl_sync(0xffffffffu, p, g))[c];
    r1 = reinterpret_cast<const uint32_t*>(__shfl_sync(0xffffffffu, p, g + 8))[c];
    r2 = reinterpret_cast<const uint32_t*>(__shfl_sync(0xffffffffu, p, g + 16))[c];
    r3 = reinterpret_cast<const uint32_t*>(__shfl_sync(0xffffffffu, p, g + 24))[c];
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s) : "memory");
#else
    const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const uintptr_t p = reinterpret_cast<uintptr_t>(smem_ptr);
    auto rd = [&](int base) -> uint32_t {
        const uint16_t lo = reinterpret_cast<const uint16_t*>(__shfl_sync(0xffffffffu, p, base + 2 * c))[g];
        const uint16_t hi = reinterpret_cast<const uint16_t*>(__shfl_sync(0xffffffffu, p, base + 2 * c + 1))[g];
        return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    };
    r0 = rd(0); r1 = rd(8); r2 = rd(16); r3 = rd(24);
#endif
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, const void* smem_ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(s) : "memory");
#else
    const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const uintptr_t p = reinterpret_cast<uintptr_t>(smem_ptr);
    r0 = reinterpret_cast<const uint32_t*>(__shfl_sync(0xffffffffu, p, g))[c];
    r1 = reinterpret_cast<const uint32_t*>(__shfl_sync(0xffffffffu, p, g + 8))[c];
#endif
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, const void* smem_ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(s) : "memory");
#else
    const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const uintptr_t p = reinterpret_cast<uintptr_t>(smem_ptr);
    auto rd = [&](int base) -> uint32_t {
        const uint16_t lo = reinterpret_cast<const uint16_t*>(__shfl_sync(0xffffffffu, p, base + 2 * c))[g];
        const uint16_t hi = reinterpret_cast<const uint16_t*>(__shfl_sync(0xffffffffu, p, base + 2 * c + 1))[g];
        return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    };
    r0 = rd(0); r1 = rd(8);
#endif
}

// ---------------------------------------------------------------------------------------------- Volta Tensor Core MMA
// Drop-in mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 backed by Volta's hardware
// mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 (4 parallel quad-pairs covering k=0..3, 4..7, 8..11, 12..15).
__device__ __forceinline__ void mma_m8n8k4_f16(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    (void)d; (void)a0; (void)a1; (void)b0; (void)b1;
#endif
}

// Out-of-line SM70 Volta Tensor Core m16n8k16 helper: keeps register pressure below Volta's 255-reg ceiling
// (preventing local-memory spills in 32-MMA unrolled tiles) and keeps ptxas compile times fast.
static __device__ __noinline__ void mma16816_f16_sm70(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                      uint32_t b0, uint32_t b1)
{
    constexpr unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    // Volta QP coordinates: r in 0..7, qp in 0..3 (qp handles k = 4*qp .. 4*qp+3)
    const int r = (lane & 3) | ((lane & 16) >> 2);
    const int qp = (lane >> 2) & 3;
    const int src0 = (r << 2) | ((qp & 1) << 1);
    const int src1 = src0 | 1;
    const bool hi_k = (qp >= 2);

    const uint32_t b0_s0 = __shfl_sync(FULL, b0, src0), b1_s0 = __shfl_sync(FULL, b1, src0);
    const uint32_t b0_s1 = __shfl_sync(FULL, b0, src1), b1_s1 = __shfl_sync(FULL, b1, src1);
    const uint32_t qb0 = hi_k ? b1_s0 : b0_s0;
    const uint32_t qb1 = hi_k ? b1_s1 : b0_s1;

    const uint32_t a0_s0 = __shfl_sync(FULL, a0, src0), a2_s0 = __shfl_sync(FULL, a2, src0);
    const uint32_t a0_s1 = __shfl_sync(FULL, a0, src1), a2_s1 = __shfl_sync(FULL, a2, src1);
    const uint32_t qa0_0 = hi_k ? a2_s0 : a0_s0;
    const uint32_t qa0_1 = hi_k ? a2_s1 : a0_s1;

    const uint32_t a1_s0 = __shfl_sync(FULL, a1, src0), a3_s0 = __shfl_sync(FULL, a3, src0);
    const uint32_t a1_s1 = __shfl_sync(FULL, a1, src1), a3_s1 = __shfl_sync(FULL, a3, src1);
    const uint32_t qa1_0 = hi_k ? a3_s0 : a1_s0;
    const uint32_t qa1_1 = hi_k ? a3_s1 : a1_s1;

    float d0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float d1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    mma_m8n8k4_f16(d0, qa0_0, qa0_1, qb0, qb1);
    mma_m8n8k4_f16(d1, qa1_0, qa1_1, qb0, qb1);

    // 4-QP reduce-scatter: each qp ends up holding its unique pair (2*k, 2*k+1), then bijective permute to m16n8k16
    const int hi4 = qp & 1;
    const int hi8 = qp >> 1;
    const int R = lane >> 2, C = lane & 3;
    const int src_d = ((R & 4) << 2) | ((R & 2) << 2) | ((C & 2) << 1) | ((C & 1) << 1) | (R & 1);

    auto reduce_and_add = [&](const float (&d)[8], float& out0, float& out1) {
        const float p0 = (hi4 ? d[4] : d[0]) + __shfl_xor_sync(FULL, hi4 ? d[0] : d[4], 4);
        const float p1 = (hi4 ? d[5] : d[1]) + __shfl_xor_sync(FULL, hi4 ? d[1] : d[5], 4);
        const float p2 = (hi4 ? d[6] : d[2]) + __shfl_xor_sync(FULL, hi4 ? d[2] : d[6], 4);
        const float p3 = (hi4 ? d[7] : d[3]) + __shfl_xor_sync(FULL, hi4 ? d[3] : d[7], 4);
        const float q0 = (hi8 ? p2 : p0) + __shfl_xor_sync(FULL, hi8 ? p0 : p2, 8);
        const float q1 = (hi8 ? p3 : p1) + __shfl_xor_sync(FULL, hi8 ? p1 : p3, 8);
        out0 += __shfl_sync(FULL, q0, src_d);
        out1 += __shfl_sync(FULL, q1, src_d);
    };
    float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
    reduce_and_add(d0, c0, c1);
    reduce_and_add(d1, c2, c3);
    c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3;
}

__device__ __forceinline__ void mma16816_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    mma16816_f16_sm70(c, a0, a1, a2, a3, b0, b1);
#endif
}

// Deferred-reduction Volta m8n8k4 accumulator for hot k-loops (tr3, ink2_tr3_moe):
// keeps partial sums in Volta QP layout across all k-tiles and runs the 12-shuffle
// reduce-scatter ONCE in flush_add(), eliminating per-tile CALL/RET, stack spills,
// and repeated output shuffles. On SM80+, compiles down to native mma16816_f16.
struct Mma16816Acc
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float c[4];
    __device__ __forceinline__ void zero() { c[0] = c[1] = c[2] = c[3] = 0.f; }
    __device__ __forceinline__ void mma(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
    {
        mma16816_f16(c, a0, a1, a2, a3, b0, b1);
    }
    __device__ __forceinline__ void flush_add(float (&out)[4]) const
    {
        out[0] += c[0]; out[1] += c[1]; out[2] += c[2]; out[3] += c[3];
    }
    __device__ __forceinline__ void flush_store(float (&out)[4]) const
    {
        out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; out[3] = c[3];
    }
#else
    float d0[8];
    float d1[8];
    __device__ __forceinline__ void zero()
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) { d0[i] = 0.f; d1[i] = 0.f; }
    }
    // Pre-shuffle A fragment (rows 0..15, k 0..15) into Volta QP layout once, reusable across NT B-tiles!
    static __device__ __forceinline__ void prep_a(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t& qa0_0, uint32_t& qa0_1, uint32_t& qa1_0, uint32_t& qa1_1)
    {
        constexpr unsigned FULL = 0xffffffffu;
        const int lane = threadIdx.x & 31;
        const int r = (lane & 3) | ((lane & 16) >> 2);
        const int qp = (lane >> 2) & 3;
        const int src0 = (r << 2) | ((qp & 1) << 1);
        const int src1 = src0 | 1;
        const bool hi_k = (qp >= 2);
        const uint32_t a0_s0 = __shfl_sync(FULL, a0, src0), a2_s0 = __shfl_sync(FULL, a2, src0);
        const uint32_t a0_s1 = __shfl_sync(FULL, a0, src1), a2_s1 = __shfl_sync(FULL, a2, src1);
        qa0_0 = hi_k ? a2_s0 : a0_s0;
        qa0_1 = hi_k ? a2_s1 : a0_s1;
        const uint32_t a1_s0 = __shfl_sync(FULL, a1, src0), a3_s0 = __shfl_sync(FULL, a3, src0);
        const uint32_t a1_s1 = __shfl_sync(FULL, a1, src1), a3_s1 = __shfl_sync(FULL, a3, src1);
        qa1_0 = hi_k ? a3_s0 : a1_s0;
        qa1_1 = hi_k ? a3_s1 : a1_s1;
    }
    // Pre-shuffle B fragment (cols 0..7, k 0..15) into Volta QP layout once, reusable across M-tiles!
    static __device__ __forceinline__ void prep_b(uint32_t b0, uint32_t b1, uint32_t& qb0, uint32_t& qb1)
    {
        constexpr unsigned FULL = 0xffffffffu;
        const int lane = threadIdx.x & 31;
        const int r = (lane & 3) | ((lane & 16) >> 2);
        const int qp = (lane >> 2) & 3;
        const int src0 = (r << 2) | ((qp & 1) << 1);
        const int src1 = src0 | 1;
        const bool hi_k = (qp >= 2);
        const uint32_t b0_s0 = __shfl_sync(FULL, b0, src0), b1_s0 = __shfl_sync(FULL, b1, src0);
        const uint32_t b0_s1 = __shfl_sync(FULL, b0, src1), b1_s1 = __shfl_sync(FULL, b1, src1);
        qb0 = hi_k ? b1_s0 : b0_s0;
        qb1 = hi_k ? b1_s1 : b0_s1;
    }
    // Row index in 0..7 for Volta QP accumulator elements d[0,1,4,5]; elements d[2,3,6,7] belong to qp_row_lo() + 2.
    // Invariant under both shfl_xor(4) and shfl_xor(8) in flush_add(), so per-row scaling commutes exactly!
    static __device__ __forceinline__ int qp_row_lo(int lane) { return ((lane & 16) >> 2) | (lane & 1); }
    __device__ __forceinline__ void add_scaled(const Mma16816Acc& g, float sc_lo, float sc_hi, float sc_lo8, float sc_hi8)
    {
        d0[0] = fmaf(sc_lo, g.d0[0], d0[0]); d0[1] = fmaf(sc_lo, g.d0[1], d0[1]);
        d0[4] = fmaf(sc_lo, g.d0[4], d0[4]); d0[5] = fmaf(sc_lo, g.d0[5], d0[5]);
        d0[2] = fmaf(sc_hi, g.d0[2], d0[2]); d0[3] = fmaf(sc_hi, g.d0[3], d0[3]);
        d0[6] = fmaf(sc_hi, g.d0[6], d0[6]); d0[7] = fmaf(sc_hi, g.d0[7], d0[7]);
        d1[0] = fmaf(sc_lo8, g.d1[0], d1[0]); d1[1] = fmaf(sc_lo8, g.d1[1], d1[1]);
        d1[4] = fmaf(sc_lo8, g.d1[4], d1[4]); d1[5] = fmaf(sc_lo8, g.d1[5], d1[5]);
        d1[2] = fmaf(sc_hi8, g.d1[2], d1[2]); d1[3] = fmaf(sc_hi8, g.d1[3], d1[3]);
        d1[6] = fmaf(sc_hi8, g.d1[6], d1[6]); d1[7] = fmaf(sc_hi8, g.d1[7], d1[7]);
    }
    __device__ __forceinline__ void mma_prepared(uint32_t qa0_0, uint32_t qa0_1, uint32_t qa1_0, uint32_t qa1_1,
                                                 uint32_t qb0, uint32_t qb1)
    {
        mma_m8n8k4_f16(d0, qa0_0, qa0_1, qb0, qb1);
        mma_m8n8k4_f16(d1, qa1_0, qa1_1, qb0, qb1);
    }
    __device__ __forceinline__ void mma(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
    {
        uint32_t qb0, qb1, qa0_0, qa0_1, qa1_0, qa1_1;
        prep_b(b0, b1, qb0, qb1);
        prep_a(a0, a1, a2, a3, qa0_0, qa0_1, qa1_0, qa1_1);
        mma_prepared(qa0_0, qa0_1, qa1_0, qa1_1, qb0, qb1);
    }
    __device__ __forceinline__ void flush_add(float (&out)[4]) const
    {
        constexpr unsigned FULL = 0xffffffffu;
        const int lane = threadIdx.x & 31;
        const int qp = (lane >> 2) & 3;
        const int hi4 = qp & 1, hi8 = qp >> 1;
        const int R = lane >> 2, C = lane & 3;
        const int src_d = ((R & 4) << 2) | ((R & 2) << 2) | ((C & 2) << 1) | ((C & 1) << 1) | (R & 1);
        auto red = [&](const float (&d)[8], float& o0, float& o1) {
            const float p0 = (hi4 ? d[4] : d[0]) + __shfl_xor_sync(FULL, hi4 ? d[0] : d[4], 4);
            const float p1 = (hi4 ? d[5] : d[1]) + __shfl_xor_sync(FULL, hi4 ? d[1] : d[5], 4);
            const float p2 = (hi4 ? d[6] : d[2]) + __shfl_xor_sync(FULL, hi4 ? d[2] : d[6], 4);
            const float p3 = (hi4 ? d[7] : d[3]) + __shfl_xor_sync(FULL, hi4 ? d[3] : d[7], 4);
            const float q0 = (hi8 ? p2 : p0) + __shfl_xor_sync(FULL, hi8 ? p0 : p2, 8);
            const float q1 = (hi8 ? p3 : p1) + __shfl_xor_sync(FULL, hi8 ? p1 : p3, 8);
            o0 += __shfl_sync(FULL, q0, src_d);
            o1 += __shfl_sync(FULL, q1, src_d);
        };
        red(d0, out[0], out[1]);
        red(d1, out[2], out[3]);
    }
    __device__ __forceinline__ void flush_store(float (&out)[4]) const
    {
        out[0] = out[1] = out[2] = out[3] = 0.f;
        flush_add(out);
    }
    // Compact out-of-line helper for high-NT (NT >= 4) prefill tiles: reuses pre-shuffled A
    // and direct-loaded B without expanding 16*NT accumulator registers.
    static __device__ __noinline__ void mma_prepared_add(float* out, uint32_t qa0_0, uint32_t qa0_1,
                                                         uint32_t qa1_0, uint32_t qa1_1, uint32_t qb0, uint32_t qb1)
    {
        Mma16816Acc tmp;
        tmp.zero();
        tmp.mma_prepared(qa0_0, qa0_1, qa1_0, qa1_1, qb0, qb1);
        float c[4] = {out[0], out[1], out[2], out[3]};
        tmp.flush_add(c);
        out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; out[3] = c[3];
    }
#endif
};

// Convert packed bf16x2 (in u32) -> packed f16x2 (in u32), clamping finite values to FP16 max (+/-65504)
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t b2)
{
    const float f0 = __uint_as_float((b2 & 0xffffu) << 16);
    const float f1 = __uint_as_float(b2 & 0xffff0000u);
    const float c0 = fminf(fmaxf(f0, -65504.0f), 65504.0f);
    const float c1 = fminf(fmaxf(f1, -65504.0f), 65504.0f);
    const __half2 h = __floats2half2_rn(c0, c1);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Drop-in mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 via Volta FP16 Tensor Cores.
__device__ __forceinline__ void mma16816_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    mma16816_f16(c, bf16x2_to_f16x2(a0), bf16x2_to_f16x2(a1), bf16x2_to_f16x2(a2), bf16x2_to_f16x2(a3),
                 bf16x2_to_f16x2(b0), bf16x2_to_f16x2(b1));
#endif
}

__device__ __forceinline__ float ue8m0_to_f32(uint32_t code)
{
    code &= 0xffu;
    return __uint_as_float(code ? (code << 23) : 0x00400000u);
}

// Out-of-line SM70 FP8/MXFP8 helpers prevent LLVM IR explosion when callers unroll 16-64x MMA loops.
static __device__ __noinline__ void mma_fp8_m16n8k32_sm70(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                          uint32_t b0, uint32_t b1)
{
    float acc[4] = {d[0], d[1], d[2], d[3]};
    mma16816_f16(acc,
                 e4m3x2_to_f16x2(a0 & 0xffffu), e4m3x2_to_f16x2(a1 & 0xffffu),
                 e4m3x2_to_f16x2(a0 >> 16),     e4m3x2_to_f16x2(a1 >> 16),
                 e4m3x2_to_f16x2(b0 & 0xffffu), e4m3x2_to_f16x2(b0 >> 16));
    mma16816_f16(acc,
                 e4m3x2_to_f16x2(a2 & 0xffffu), e4m3x2_to_f16x2(a3 & 0xffffu),
                 e4m3x2_to_f16x2(a2 >> 16),     e4m3x2_to_f16x2(a3 >> 16),
                 e4m3x2_to_f16x2(b1 & 0xffffu), e4m3x2_to_f16x2(b1 >> 16));
    d[0] = acc[0]; d[1] = acc[1]; d[2] = acc[2]; d[3] = acc[3];
}

// Drop-in mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 via 2x Volta FP16 m16n8k16.
// Inner product over the 32 k-elements is invariant to identical k-permutation on A and B.
__device__ __forceinline__ void mma_fp8_m16n8k32(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    mma_fp8_m16n8k32_sm70(d, a0, a1, a2, a3, b0, b1);
#endif
}

static __device__ __noinline__ void mma_mx8_m16n8k32_sm70(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                          uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb)
{
    float t[4] = {0.f, 0.f, 0.f, 0.f};
    mma_fp8_m16n8k32_sm70(t, a0, a1, a2, a3, b0, b1);
    const int lane = threadIdx.x & 31;
    const int R = lane >> 2, C = lane & 3;
    const float s_alo = ue8m0_to_f32(__shfl_sync(0xffffffffu, sa, R * 4));
    const float s_ahi = ue8m0_to_f32(__shfl_sync(0xffffffffu, sa, R * 4 + 1));
    const float s_b0  = ue8m0_to_f32(__shfl_sync(0xffffffffu, sb, (2 * C) * 4));
    const float s_b1  = ue8m0_to_f32(__shfl_sync(0xffffffffu, sb, (2 * C + 1) * 4));
    d[0] += t[0] * (s_alo * s_b0);
    d[1] += t[1] * (s_alo * s_b1);
    d[2] += t[2] * (s_ahi * s_b0);
    d[3] += t[3] * (s_ahi * s_b1);
}

// Drop-in mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
__device__ __forceinline__ void mma_mx8_m16n8k32(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const uint16_t z = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
#else
    mma_mx8_m16n8k32_sm70(d, a0, a1, a2, a3, b0, b1, sa, sb);
#endif
}

#endif   // __CUDACC__
}   // namespace gr8_sm70
