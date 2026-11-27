// Unit test for SM70 (Volta) compatibility primitives (sm70_compat.cuh + sm70_cublas.cuh).
// Exercises all FP8/FP4 conversions, ldmatrix.x{2,4}[.trans], Volta hardware Tensor Core
// mma16816_f16 / mma16816_bf16 / mma_fp8_m16n8k32 / mma_mx8_m16n8k32, and cublas_gemm_bf16_tn.
//
// Build: nvcc -O2 -std=c++17 -DGR8_SM70=1 -gencode=arch=compute_70,code=sm_70 \
//            -o bin/sm70_mma_test engine/kernels/tests/sm70_mma_test.cu -lcublas
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include "../sm70_compat.cuh"
#include "../sm70_cublas.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// 1. Exhaustive 256-byte E4M3 and E2M1 conversion checks
__global__ void k_test_cvt(uint16_t* e4_out, uint16_t* e2_lo, uint16_t* e2_hi, uint32_t* h2_out)
{
    const int i = threadIdx.x;
    if (i < 256)
    {
        const uint32_t p = gr8_sm70::e4m3x2_to_f16x2(static_cast<uint32_t>(i) | (static_cast<uint32_t>(i ^ 0x55) << 8));
        e4_out[i] = static_cast<uint16_t>(p & 0xffffu);
        const uint32_t q = gr8_sm70::e2m1x2_to_f16x2(static_cast<uint32_t>(i));
        e2_lo[i] = static_cast<uint16_t>(q & 0xffffu);
        e2_hi[i] = static_cast<uint16_t>(q >> 16);
    }
    if (i == 0)
    {
        const __half2 a = __floats2half2_rn(-3.5f, 1.25f);
        const __half2 b = __floats2half2_rn(2.0f, -4.0f);
        const __half2 ha = gr8_sm70::habs2(a);
        const __half2 hm = gr8_sm70::hmax2(ha, gr8_sm70::habs2(b));
        h2_out[0] = *reinterpret_cast<const uint32_t*>(&ha);
        h2_out[1] = *reinterpret_cast<const uint32_t*>(&hm);
    }
}

// 1b. Exhaustive 65,536-half check for fast inline f16x2_to_e4m3x2 & round_f16x2_e4m3
__global__ void k_test_f16_e4m3_all(uint8_t* out_e4, uint16_t* out_rnd, uint16_t* out_bnd)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 32768) return;
    const uint32_t h2 = static_cast<uint32_t>(2 * idx) | (static_cast<uint32_t>(2 * idx + 1) << 16);
    const uint16_t e2 = gr8_sm70::f16x2_to_e4m3x2(h2);
    out_e4[2 * idx] = static_cast<uint8_t>(e2 & 0xffu);
    out_e4[2 * idx + 1] = static_cast<uint8_t>(e2 >> 8);
    const uint32_t r2 = gr8_sm70::round_f16x2_e4m3(h2);
    out_rnd[2 * idx] = static_cast<uint16_t>(r2 & 0xffffu);
    out_rnd[2 * idx + 1] = static_cast<uint16_t>(r2 >> 16);
    const uint32_t b2 = gr8_sm70::round_f16x2_e4m3_bounded(h2);
    out_bnd[2 * idx] = static_cast<uint16_t>(b2 & 0xffffu);
    out_bnd[2 * idx + 1] = static_cast<uint16_t>(b2 >> 16);
}

// 2. ldmatrix x4 / x4.trans / x2 / x2.trans checks (16x16 u16 tile in shared memory)
__global__ void k_test_ldmatrix(const uint16_t* tile, uint32_t* out_x4, uint32_t* out_x4t, uint32_t* out_x2, uint32_t* out_x2t)
{
    __shared__ uint16_t sm[256];
    for (int i = threadIdx.x; i < 256; i += 32) sm[i] = tile[i];
    __syncthreads();
    const int lane = threadIdx.x;
    // 4 tiles of 8x8: tile 0 = rows 0..7 cols 0..7, tile 1 = rows 8..15 cols 0..7,
    //                 tile 2 = rows 0..7 cols 8..15, tile 3 = rows 8..15 cols 8..15
    const int row = (lane & 7) + ((lane & 8) ? 8 : 0);
    const int col = (lane & 16) ? 8 : 0;
    const uint16_t* ptr = sm + row * 16 + col;
    uint32_t r0, r1, r2, r3;
    gr8_sm70::ldmatrix_x4(r0, r1, r2, r3, ptr);
    out_x4[lane * 4 + 0] = r0; out_x4[lane * 4 + 1] = r1; out_x4[lane * 4 + 2] = r2; out_x4[lane * 4 + 3] = r3;

    gr8_sm70::ldmatrix_x4_trans(r0, r1, r2, r3, ptr);
    out_x4t[lane * 4 + 0] = r0; out_x4t[lane * 4 + 1] = r1; out_x4t[lane * 4 + 2] = r2; out_x4t[lane * 4 + 3] = r3;

    gr8_sm70::ldmatrix_x2(r0, r1, ptr);
    out_x2[lane * 2 + 0] = r0; out_x2[lane * 2 + 1] = r1;

    gr8_sm70::ldmatrix_x2_trans(r0, r1, ptr);
    out_x2t[lane * 2 + 0] = r0; out_x2t[lane * 2 + 1] = r1;
}

// 3. mma16816_f16 & mma16816_bf16: A[16][16] row-major, B[8][16] row-major (i.e. B^T is 16x8 col-major), C[16][8]
__global__ void k_test_mma16(const uint16_t* A, const uint16_t* B, const float* Cin, float* Cout, int is_bf16)
{
    const int lane = threadIdx.x;
    const int g = lane >> 2, c = lane & 3;
    auto pack2 = [](const uint16_t* p) -> uint32_t {
        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 16);
    };
    const uint32_t a0 = pack2(A + g * 16 + 2 * c);
    const uint32_t a1 = pack2(A + (g + 8) * 16 + 2 * c);
    const uint32_t a2 = pack2(A + g * 16 + 2 * c + 8);
    const uint32_t a3 = pack2(A + (g + 8) * 16 + 2 * c + 8);
    const uint32_t b0 = pack2(B + g * 16 + 2 * c);
    const uint32_t b1 = pack2(B + g * 16 + 2 * c + 8);

    float d[4] = {
        Cin[g * 8 + 2 * c],
        Cin[g * 8 + 2 * c + 1],
        Cin[(g + 8) * 8 + 2 * c],
        Cin[(g + 8) * 8 + 2 * c + 1]
    };
    if (is_bf16)
        gr8_sm70::mma16816_bf16(d, a0, a1, a2, a3, b0, b1);
    else
        gr8_sm70::mma16816_f16(d, a0, a1, a2, a3, b0, b1);

    Cout[g * 8 + 2 * c]           = d[0];
    Cout[g * 8 + 2 * c + 1]       = d[1];
    Cout[(g + 8) * 8 + 2 * c]     = d[2];
    Cout[(g + 8) * 8 + 2 * c + 1] = d[3];
}

// 4. mma_fp8_m16n8k32 & mma_mx8_m16n8k32: A[16][32] E4M3, B[8][32] E4M3, scales sa[16], sb[8]
__global__ void k_test_mma_fp8(const uint8_t* A, const uint8_t* B, const uint8_t* sa_arr, const uint8_t* sb_arr,
                               const float* Cin, float* Cout, int use_mx)
{
    const int lane = threadIdx.x;
    const int g = lane >> 2, c = lane & 3;
    auto pack4 = [](const uint8_t* p) -> uint32_t {
        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    };
    const uint32_t a0 = pack4(A + g * 32 + 4 * c);
    const uint32_t a1 = pack4(A + (g + 8) * 32 + 4 * c);
    const uint32_t a2 = pack4(A + g * 32 + 4 * c + 16);
    const uint32_t a3 = pack4(A + (g + 8) * 32 + 4 * c + 16);
    const uint32_t b0 = pack4(B + g * 32 + 4 * c);
    const uint32_t b1 = pack4(B + g * 32 + 4 * c + 16);

    float d[4] = {
        Cin[g * 8 + 2 * c],
        Cin[g * 8 + 2 * c + 1],
        Cin[(g + 8) * 8 + 2 * c],
        Cin[(g + 8) * 8 + 2 * c + 1]
    };
    if (use_mx)
    {
        const uint32_t sa = sa_arr[g + ((lane & 1) ? 8 : 0)];
        const uint32_t sb = sb_arr[g];
        gr8_sm70::mma_mx8_m16n8k32(d, a0, a1, a2, a3, b0, b1, sa, sb);
    }
    else
    {
        gr8_sm70::mma_fp8_m16n8k32(d, a0, a1, a2, a3, b0, b1);
    }

    Cout[g * 8 + 2 * c]           = d[0];
    Cout[g * 8 + 2 * c + 1]       = d[1];
    Cout[(g + 8) * 8 + 2 * c]     = d[2];
    Cout[(g + 8) * 8 + 2 * c + 1] = d[3];
}

static uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
static float randf(uint32_t& s) { return (static_cast<float>(lcg(s) >> 8) * (1.f / 16777216.f) - 0.5f) * 2.f; }

int main()
{
    int dev = 0;
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("sm70_mma_test on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // ---- 1. E4M3 & E2M1 conversions
    uint16_t *d_e4, *d_e2lo, *d_e2hi;
    uint32_t *d_h2;
    CK(cudaMalloc(&d_e4, 256 * 2));
    CK(cudaMalloc(&d_e2lo, 256 * 2));
    CK(cudaMalloc(&d_e2hi, 256 * 2));
    CK(cudaMalloc(&d_h2, 8));
    k_test_cvt<<<1, 256>>>(d_e4, d_e2lo, d_e2hi, d_h2);
    CK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_e4(256), h_e2lo(256), h_e2hi(256);
    uint32_t h_h2[2];
    CK(cudaMemcpy(h_e4.data(), d_e4, 512, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_e2lo.data(), d_e2lo, 512, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_e2hi.data(), d_e2hi, 512, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_h2, d_h2, 8, cudaMemcpyDeviceToHost));

    int bad_e4 = 0, bad_e2 = 0;
    for (int i = 0; i < 256; ++i)
    {
        const uint16_t want = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(i), __NV_E4M3).x;
        if (h_e4[i] != want) { if (bad_e4 < 5) fprintf(stderr, "e4m3[%#x]: got %#x want %#x\n", i, h_e4[i], want); ++bad_e4; }
    }
    const float e2_lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    for (int i = 0; i < 256; ++i)
    {
        auto want_nib = [&](int n) -> uint16_t {
            float v = e2_lut[n & 7];
            uint16_t b = __half_raw(__float2half_rn(v)).x;
            return (n & 8) ? static_cast<uint16_t>(b | 0x8000u) : b;
        };
        if (h_e2lo[i] != want_nib(i & 0xf) || h_e2hi[i] != want_nib(i >> 4)) ++bad_e2;
    }
    const __half2 h_ha = *reinterpret_cast<const __half2*>(&h_h2[0]);
    const __half2 h_hm = *reinterpret_cast<const __half2*>(&h_h2[1]);
    if (bad_e4 || bad_e2 || __half2float(__low2half(h_ha)) != 3.5f || __half2float(__high2half(h_ha)) != 1.25f ||
        __half2float(__low2half(h_hm)) != 3.5f || __half2float(__high2half(h_hm)) != 4.0f)
    {
        fprintf(stderr, "FAIL cvt: bad_e4=%d bad_e2=%d\n", bad_e4, bad_e2);
        return 1;
    }
    printf("PASS cvt (256 E4M3 + 256 E2M1x2 + habs2/hmax2 bit-exact)\n");

    // ---- 1b. Exhaustive 65,536-half f16x2_to_e4m3x2 & round_f16x2_e4m3
    uint8_t* d_fe4;
    uint16_t *d_frnd, *d_fbnd;
    CK(cudaMalloc(&d_fe4, 65536));
    CK(cudaMalloc(&d_frnd, 65536 * 2));
    CK(cudaMalloc(&d_fbnd, 65536 * 2));
    k_test_f16_e4m3_all<<<128, 256>>>(d_fe4, d_frnd, d_fbnd);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> h_fe4(65536);
    std::vector<uint16_t> h_frnd(65536), h_fbnd(65536);
    CK(cudaMemcpy(h_fe4.data(), d_fe4, 65536, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_frnd.data(), d_frnd, 65536 * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_fbnd.data(), d_fbnd, 65536 * 2, cudaMemcpyDeviceToHost));
    int bad_f2e = 0, bad_rnd = 0, bad_bnd = 0;
    for (int i = 0; i < 65536; ++i)
    {
        __half_raw hr; hr.x = static_cast<unsigned short>(i);
        const uint8_t want_e = static_cast<uint8_t>(__nv_cvt_halfraw_to_fp8(hr, __NV_SATFINITE, __NV_E4M3));
        const uint16_t want_r = __nv_cvt_fp8_to_halfraw(want_e, __NV_E4M3).x;
        if (h_fe4[i] != want_e) { if (bad_f2e < 8) fprintf(stderr, "f16->e4m3[%#06x]: got %#04x want %#04x\n", i, h_fe4[i], want_e); ++bad_f2e; }
        if (h_frnd[i] != want_r) { if (bad_rnd < 8) fprintf(stderr, "round_e4m3[%#06x]: got %#06x want %#06x\n", i, h_frnd[i], want_r); ++bad_rnd; }
        if ((i & 0x7fff) < 0x5f00 && h_fbnd[i] != want_r) { if (bad_bnd < 8) fprintf(stderr, "round_bnd[%#06x]: got %#06x want %#06x\n", i, h_fbnd[i], want_r); ++bad_bnd; }
    }
    if (bad_f2e || bad_rnd || bad_bnd)
    {
        fprintf(stderr, "FAIL f16_e4m3_all: bad_f2e=%d bad_rnd=%d bad_bnd=%d\n", bad_f2e, bad_rnd, bad_bnd);
        return 1;
    }
    printf("PASS f16_e4m3_all (all 65536 halfs bit-exact for f16->e4m3, round_e4m3, round_bounded)\n");

    // ---- 2. ldmatrix
    std::vector<uint16_t> tile(256);
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            tile[r * 16 + c] = static_cast<uint16_t>((r << 8) | c);
    uint16_t* d_tile;
    uint32_t *d_x4, *d_x4t, *d_x2, *d_x2t;
    CK(cudaMalloc(&d_tile, 512));
    CK(cudaMalloc(&d_x4, 128 * 4)); CK(cudaMalloc(&d_x4t, 128 * 4));
    CK(cudaMalloc(&d_x2, 64 * 4));  CK(cudaMalloc(&d_x2t, 64 * 4));
    CK(cudaMemcpy(d_tile, tile.data(), 512, cudaMemcpyHostToDevice));
    k_test_ldmatrix<<<1, 32>>>(d_tile, d_x4, d_x4t, d_x2, d_x2t);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> hx4(128), hx4t(128), hx2(64), hx2t(64);
    CK(cudaMemcpy(hx4.data(), d_x4, 512, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hx4t.data(), d_x4t, 512, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hx2.data(), d_x2, 256, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hx2t.data(), d_x2t, 256, cudaMemcpyDeviceToHost));
    int bad_ld = 0;
    for (int lane = 0; lane < 32; ++lane)
    {
        const int g = lane >> 2, c = lane & 3;
        const int r_off[4] = {0, 8, 0, 8}, c_off[4] = {0, 0, 8, 8};
        for (int m = 0; m < 4; ++m)
        {
            const uint32_t w_n = static_cast<uint32_t>(tile[(r_off[m] + g) * 16 + c_off[m] + 2 * c]) |
                                 (static_cast<uint32_t>(tile[(r_off[m] + g) * 16 + c_off[m] + 2 * c + 1]) << 16);
            const uint32_t w_t = static_cast<uint32_t>(tile[(r_off[m] + 2 * c) * 16 + c_off[m] + g]) |
                                 (static_cast<uint32_t>(tile[(r_off[m] + 2 * c + 1) * 16 + c_off[m] + g]) << 16);
            if (hx4[lane * 4 + m] != w_n) ++bad_ld;
            if (hx4t[lane * 4 + m] != w_t) ++bad_ld;
            if (m < 2 && hx2[lane * 2 + m] != w_n) ++bad_ld;
            if (m < 2 && hx2t[lane * 2 + m] != w_t) ++bad_ld;
        }
    }
    if (bad_ld) { fprintf(stderr, "FAIL ldmatrix: %d mismatches\n", bad_ld); return 1; }
    printf("PASS ldmatrix (x4, x4.trans, x2, x2.trans exact)\n");

    // ---- 3. mma16816_f16 & mma16816_bf16
    uint32_t seed = 0x12345678u;
    uint16_t *d_A16, *d_B16;
    float *d_Cin, *d_Cout;
    CK(cudaMalloc(&d_A16, 16 * 16 * 2));
    CK(cudaMalloc(&d_B16, 8 * 16 * 2));
    CK(cudaMalloc(&d_Cin, 16 * 8 * 4));
    CK(cudaMalloc(&d_Cout, 16 * 8 * 4));
    for (int is_bf16 = 0; is_bf16 < 2; ++is_bf16)
    {
        float max_err = 0.f;
        for (int trial = 0; trial < 8; ++trial)
        {
            std::vector<uint16_t> hA(16 * 16), hB(8 * 16);
            std::vector<float> fA(16 * 16), fB(8 * 16), hCin(16 * 8), hCout(16 * 8);
            for (int i = 0; i < 16 * 16; ++i)
            {
                float v = randf(seed) * 4.f;
                if (is_bf16) { __nv_bfloat16 b = __float2bfloat16_rn(v); hA[i] = __nv_bfloat16_raw(b).x; fA[i] = __bfloat162float(b); }
                else         { __half h = __float2half_rn(v);            hA[i] = __half_raw(h).x;        fA[i] = __half2float(h); }
            }
            for (int i = 0; i < 8 * 16; ++i)
            {
                float v = randf(seed) * 4.f;
                if (is_bf16) { __nv_bfloat16 b = __float2bfloat16_rn(v); hB[i] = __nv_bfloat16_raw(b).x; fB[i] = __bfloat162float(b); }
                else         { __half h = __float2half_rn(v);            hB[i] = __half_raw(h).x;        fB[i] = __half2float(h); }
            }
            for (int i = 0; i < 16 * 8; ++i) hCin[i] = randf(seed) * 2.f;
            CK(cudaMemcpy(d_A16, hA.data(), 512, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_B16, hB.data(), 256, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_Cin, hCin.data(), 512, cudaMemcpyHostToDevice));
            k_test_mma16<<<1, 32>>>(d_A16, d_B16, d_Cin, d_Cout, is_bf16);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hCout.data(), d_Cout, 512, cudaMemcpyDeviceToHost));
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 8; ++c)
                {
                    double ref = hCin[r * 8 + c];
                    for (int k = 0; k < 16; ++k) ref += static_cast<double>(fA[r * 16 + k]) * static_cast<double>(fB[c * 16 + k]);
                    max_err = fmaxf(max_err, fabsf(hCout[r * 8 + c] - static_cast<float>(ref)));
                }
        }
        const float tol = is_bf16 ? 1e-2f : 1e-4f;
        if (max_err > tol) { fprintf(stderr, "FAIL mma16816_%s: max_err=%g\n", is_bf16 ? "bf16" : "f16", max_err); return 1; }
        printf("PASS mma16816_%s (max_err=%g <= %g)\n", is_bf16 ? "bf16" : "f16", max_err, tol);
    }

    // ---- 4. mma_fp8_m16n8k32 & mma_mx8_m16n8k32
    uint8_t *d_A8, *d_B8, *d_sa, *d_sb;
    CK(cudaMalloc(&d_A8, 16 * 32));
    CK(cudaMalloc(&d_B8, 8 * 32));
    CK(cudaMalloc(&d_sa, 16));
    CK(cudaMalloc(&d_sb, 8));
    for (int use_mx = 0; use_mx < 2; ++use_mx)
    {
        float max_err = 0.f;
        for (int trial = 0; trial < 8; ++trial)
        {
            std::vector<uint8_t> hA(16 * 32), hB(8 * 32), hsa(16), hsb(8);
            std::vector<float> fA(16 * 32), fB(8 * 32), hCin(16 * 8), hCout(16 * 8);
            for (int i = 0; i < 16 * 32; ++i)
            {
                uint8_t b = static_cast<uint8_t>(lcg(seed) >> 24);
                if ((b & 0x7fu) == 0x7fu) b = 0x38u; // avoid NaN
                if ((b & 0x78u) >= 0x58u) b &= ~0x20u; // keep magnitude moderate
                hA[i] = b;
                fA[i] = __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
            }
            for (int i = 0; i < 8 * 32; ++i)
            {
                uint8_t b = static_cast<uint8_t>(lcg(seed) >> 24);
                if ((b & 0x7fu) == 0x7fu) b = 0x38u;
                if ((b & 0x78u) >= 0x58u) b &= ~0x20u;
                hB[i] = b;
                fB[i] = __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
            }
            for (int i = 0; i < 16; ++i) hsa[i] = static_cast<uint8_t>(124 + (lcg(seed) % 7));
            for (int i = 0; i < 8; ++i)  hsb[i] = static_cast<uint8_t>(124 + (lcg(seed) % 7));
            for (int i = 0; i < 16 * 8; ++i) hCin[i] = randf(seed) * 2.f;
            CK(cudaMemcpy(d_A8, hA.data(), 512, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_B8, hB.data(), 256, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_sa, hsa.data(), 16, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_sb, hsb.data(), 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_Cin, hCin.data(), 512, cudaMemcpyHostToDevice));
            k_test_mma_fp8<<<1, 32>>>(d_A8, d_B8, d_sa, d_sb, d_Cin, d_Cout, use_mx);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hCout.data(), d_Cout, 512, cudaMemcpyDeviceToHost));
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 8; ++c)
                {
                    double dot = 0.0;
                    for (int k = 0; k < 32; ++k) dot += static_cast<double>(fA[r * 32 + k]) * static_cast<double>(fB[c * 32 + k]);
                    if (use_mx) dot *= ldexp(1.0, static_cast<int>(hsa[r]) - 127) * ldexp(1.0, static_cast<int>(hsb[c]) - 127);
                    const double ref = hCin[r * 8 + c] + dot;
                    max_err = fmaxf(max_err, fabsf(hCout[r * 8 + c] - static_cast<float>(ref)));
                }
        }
        if (max_err > 1e-2f) { fprintf(stderr, "FAIL mma_%s: max_err=%g\n", use_mx ? "mx8" : "fp8", max_err); return 1; }
        printf("PASS mma_%s_m16n8k32 (max_err=%g)\n", use_mx ? "mx8" : "fp8", max_err);
    }

    // ---- 5. cublas_gemm_bf16_tn
    cublasHandle_t bh;
    if (cublasCreate(&bh) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 1; }
    const int M = 17, N = 33, K = 64;
    std::vector<__nv_bfloat16> bA(N * K), bB(M * K), bC(M * N);
    std::vector<float> fC(M * N);
    for (int i = 0; i < N * K; ++i) bA[i] = __float2bfloat16_rn(randf(seed));
    for (int i = 0; i < M * K; ++i) bB[i] = __float2bfloat16_rn(randf(seed));
    __nv_bfloat16 *dbA, *dbB, *dbC;
    float* dfC;
    CK(cudaMalloc(&dbA, N * K * 2)); CK(cudaMalloc(&dbB, M * K * 2));
    CK(cudaMalloc(&dbC, M * N * 2)); CK(cudaMalloc(&dfC, M * N * 4));
    CK(cudaMemcpy(dbA, bA.data(), N * K * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbB, bB.data(), M * K * 2, cudaMemcpyHostToDevice));
    const float alpha = 1.f, beta = 0.f;
    if (gr8_sm70::cublas_gemm_bf16_tn(bh, N, M, K, &alpha, dbA, K, dbB, K, &beta, dfC, CUDA_R_32F, N) != CUBLAS_STATUS_SUCCESS ||
        gr8_sm70::cublas_gemm_bf16_tn(bh, N, M, K, &alpha, dbA, K, dbB, K, &beta, dbC, CUDA_R_16BF, N) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "FAIL cublas_gemm_bf16_tn call failed\n");
        return 1;
    }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(fC.data(), dfC, M * N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bC.data(), dbC, M * N * 2, cudaMemcpyDeviceToHost));
    float max_f = 0.f, max_b = 0.f;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
        {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) ref += static_cast<double>(__bfloat162float(bA[n * K + k])) * static_cast<double>(__bfloat162float(bB[m * K + k]));
            max_f = fmaxf(max_f, fabsf(fC[m * N + n] - static_cast<float>(ref)));
            max_b = fmaxf(max_b, fabsf(__bfloat162float(bC[m * N + n]) - static_cast<float>(ref)));
        }
    if (max_f > 2e-2f || max_b > 5e-2f) { fprintf(stderr, "FAIL cublas_gemm_bf16_tn: f=%g b=%g\n", max_f, max_b); return 1; }
    printf("PASS cublas_gemm_bf16_tn (fp32 err=%g, bf16 err=%g)\n", max_f, max_b);
    printf("sm70_mma_test: ALL PASS\n");
    return 0;
}
