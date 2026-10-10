// Prefill GEMM with real memory traffic, one kernel per operand type.
//   f16 / bf16  m16n8k16      fp8 e4m3  m16n8k32      nvfp4 (e2m1 + ue4m3 scale per 16 k)  m16n8k64
//   tr3fp8: B decoded from TR3 tiles and rounded to e4m3 in registers, A = FP8 activations, m16n8k32.
// Y[rows, N] (fp16) = X[rows, K] @ W[K, N] * alpha.  CTA = (128 output cols, <= MB*16 rows of one pool entry),
// 8 warps, one 16-col tile per warp.  Operands are stored fragment-ready:
//   X  : [row block of 16][k step][lane][a0 a1 a2 a3]          16 B per lane, streamed through smem (cp.async)
//   W  : [k step][16-col tile][lane][b0 b1 | b0 b1]            cols 0-7 | cols 8-15, read straight into registers
//   nvfp4 scales: XS [row block][k step][16 rows] u32, WS [k step][tile][16 cols] u32 (4 ue4m3, one per 16 k)
// tr3fp8 k order inside a k32 step: slot 4c+i of reg pair h = natural k 16*(i>>1) + 2c + (i&1) + 8h, i.e. the
// X->FP8 pass writes k in the order the two trellis tiles deliver it (no shuffles in the kernel).
#pragma once
#include "tr3_gemm.cuh"

namespace gx
{
enum { F16 = 0, BF16, FP8, NVF4, TR3FP8, KINDS };
constexpr int STAGES = 4, KS = 2;
constexpr int KSZ[KINDS] = {16, 16, 32, 64, 32};

template <int KIND>
__device__ __forceinline__ void mma(float (&d)[4], const uint4& a, u32 b0, u32 b1, u32 sa, u32 sb)
{
    if constexpr (KIND == F16)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
    else if constexpr (KIND == BF16)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
    else if constexpr (KIND == NVF4)
    {
        // scale data (thread ids 0): quad lane 0 supplies row l/4, lane 1 row l/4+8; quad lane 0 supplies col l/4
        const uint16_t z = 0;
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
    }
    else
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ u32 h2_e4m3x2(u32 h2)   // half2 -> two e4m3 bytes (low half = low byte), RN, saturating
{
    const float2 f = __half22float2(as_h2(h2));
    u32 r;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; cvt.u32.u16 %0, t; }" : "=r"(r) : "f"(f.x), "f"(f.y));
    return r;
}

template <int KIND, int MB, int BITS>
__device__ __forceinline__ void body(const uint8_t* __restrict__ X, const u32* __restrict__ XS, const uint8_t* __restrict__ W,
                                     const u32* __restrict__ WS, __half* __restrict__ Y, int S, int N, int rb0, int nb, float alpha,
                                     unsigned char* smem)
{
    constexpr int TB = 32 * BITS;
    constexpr int A_STAGE = KS * MB * 512;
    constexpr int CHUNKS = A_STAGE / 16;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int NT16 = N / 16, tile = blockIdx.x * 8 + warp;
    const int NST = S / KS;

    auto loadA = [&](int slot, int st)
    {
        for (int c = t; c < CHUNKS; c += 256)
        {
            const int j = c / (MB * 32), mb = (c >> 5) % MB, l = c & 31;
            if (mb < nb)
                pf::cp16(smem + slot * A_STAGE + c * 16, X + (static_cast<size_t>(rb0 + mb) * S + st * KS + j) * 512 + l * 16);
        }
    };
    Tr3Lane<BITS> ld;
    ld.init(lane);
    auto loadB = [&](int ks, u32 (&v)[4], u32 (&sc)[2])
    {
        sc[0] = sc[1] = 0;
        if constexpr (KIND == TR3FP8)
        {
            const uint8_t* p = W + (static_cast<size_t>(2 * ks) * NT16 + tile) * TB;
            ld.load(p, v[0], v[1]);
            ld.load(p + static_cast<size_t>(NT16) * TB, v[2], v[3]);
        }
        else
        {
            const uint4 q = *reinterpret_cast<const uint4*>(W + (static_cast<size_t>(ks) * NT16 + tile) * 512 + lane * 16);
            v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
            if constexpr (KIND == NVF4)
            {
                const u32* s = WS + (static_cast<size_t>(ks) * NT16 + tile) * 16 + (lane >> 2);
                sc[0] = s[0]; sc[1] = s[8];
            }
        }
    };
    auto frags = [&](const u32 (&v)[4], u32 (&f0)[2], u32 (&f1)[2])
    {
        if constexpr (KIND == TR3FP8)
        {
            u32 p0[2], p1[2], q0[2], q1[2];
            ld.decode(v[0], v[1], p0, p1);
            ld.decode(v[2], v[3], q0, q1);
            f0[0] = h2_e4m3x2(p0[0]) | (h2_e4m3x2(q0[0]) << 16);
            f0[1] = h2_e4m3x2(p0[1]) | (h2_e4m3x2(q0[1]) << 16);
            f1[0] = h2_e4m3x2(p1[0]) | (h2_e4m3x2(q1[0]) << 16);
            f1[1] = h2_e4m3x2(p1[1]) | (h2_e4m3x2(q1[1]) << 16);
        }
        else { f0[0] = v[0]; f0[1] = v[1]; f1[0] = v[2]; f1[1] = v[3]; }
    };

    #pragma unroll
    for (int p = 0; p < STAGES - 1; ++p)
    {
        if (p < NST) loadA(p, p);
        pf::cp_commit();
    }
    u32 nv[KS][4], ns[KS][2];
    #pragma unroll
    for (int j = 0; j < KS; ++j) loadB(j, nv[j], ns[j]);
    float acc[MB][2][4];
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mb][h][i] = 0.f;
    const int srow = (lane >> 2) + ((lane & 1) ? 8 : 0);   // probed on sm_120: thread-id-a 0 -> quad lane 0 gives row l/4, lane 1 row l/4+8

    for (int st = 0; st < NST; ++st)
    {
        pf::cp_wait<STAGES - 2>();
        __syncthreads();
        if (st + STAGES - 1 < NST) loadA((st + STAGES - 1) % STAGES, st + STAGES - 1);
        pf::cp_commit();
        u32 cv[KS][4], cs[KS][2];
        #pragma unroll
        for (int j = 0; j < KS; ++j)
        {
            #pragma unroll
            for (int i = 0; i < 4; ++i) cv[j][i] = nv[j][i];
            cs[j][0] = ns[j][0]; cs[j][1] = ns[j][1];
        }
        if (st + 1 < NST)
        {
            #pragma unroll
            for (int j = 0; j < KS; ++j) loadB((st + 1) * KS + j, nv[j], ns[j]);
        }
        u32 sa[KS][MB];
        #pragma unroll
        for (int j = 0; j < KS; ++j)
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                sa[j][mb] = 0;
                if constexpr (KIND == NVF4)
                    if (mb < nb) sa[j][mb] = XS[(static_cast<size_t>(rb0 + mb) * S + st * KS + j) * 16 + srow];
            }
        const unsigned char* a = smem + (st % STAGES) * A_STAGE;
        #pragma unroll
        for (int j = 0; j < KS; ++j)
        {
            u32 f0[2], f1[2];
            frags(cv[j], f0, f1);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                if (mb < nb)
                {
                    const uint4 fa = *reinterpret_cast<const uint4*>(a + (j * MB + mb) * 512 + lane * 16);
                    mma<KIND>(acc[mb][0], fa, f0[0], f0[1], sa[j][mb], cs[j][0]);
                    mma<KIND>(acc[mb][1], fa, f1[0], f1[1], sa[j][mb], cs[j][1]);
                }
            }
        }
    }
    pf::cp_wait<0>();

    const int c = lane & 3, r = lane >> 2;
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            if (mb >= nb) continue;
            const int col = tile * 16 + h * 8 + 2 * c;
            const size_t row = static_cast<size_t>(rb0 + mb) * 16 + r;
            *reinterpret_cast<u32*>(Y + row * N + col) = pack_h2(acc[mb][h][0] * alpha, acc[mb][h][1] * alpha);
            *reinterpret_cast<u32*>(Y + (row + 8) * N + col) = pack_h2(acc[mb][h][2] * alpha, acc[mb][h][3] * alpha);
        }
}
}  // namespace gx

// segs: [nseg][3] = (pool entry, first row block, row blocks <= MB)
template <int KIND, int MB>
__global__ __launch_bounds__(256) void gx_gemm(const uint8_t* X, const u32* XS, const uint8_t* W, const u32* WS, size_t wstride,
                                               size_t wsstride, int bits, const int* segs, __half* Y, int S, int N, float alpha)
{
    extern __shared__ __align__(16) unsigned char smem_gx[];
    const int* sg = segs + blockIdx.y * 3;
    const uint8_t* w = W + sg[0] * wstride;
    const u32* ws = WS + sg[0] * wsstride;
    if constexpr (KIND == gx::TR3FP8)
    {
        if (bits == 3) gx::body<KIND, MB, 3>(X, XS, w, ws, Y, S, N, sg[1], sg[2], alpha, smem_gx);
        else gx::body<KIND, MB, 4>(X, XS, w, ws, Y, S, N, sg[1], sg[2], alpha, smem_gx);
    }
    else gx::body<KIND, MB, 3>(X, XS, w, ws, Y, S, N, sg[1], sg[2], alpha, smem_gx);
}

#ifdef HOST_CMDS
template <int KIND, int MB>
static void gx_launch(dim3 grid, const uint8_t* X, const u32* XS, const uint8_t* W, const u32* WS, size_t wstride, size_t wsstride, int bits,
                      const int* segs, __half* Y, int S, int N, float alpha)
{
    gx_gemm<KIND, MB><<<grid, 256, gx::STAGES * gx::KS * MB * 512>>>(X, XS, W, WS, wstride, wsstride, bits, segs, Y, S, N, alpha);
}

static int cmd_gemm(const Args& a)
{
    static const char* const names[gx::KINDS] = {"f16", "bf16", "fp8", "nvfp4", "tr3fp8"};
    const std::string kname = a.s("kind", "fp8");
    int kind = -1;
    for (int i = 0; i < gx::KINDS; ++i) if (kname == names[i]) kind = i;
    const int K = a.i("K", 6144), N = a.i("N", 768), bits = a.i("bits", 3), MB = a.i("MB", 4);
    const int E = a.i("experts", 64), rpe = a.i("rpe", 64), P = a.i("pool", 16), reps = a.i("reps", 5);
    const std::string dump = a.s("dump", "");
    if (kind < 0 || (MB != 4 && MB != 8)) { fprintf(stderr, "bad kind/MB\n"); return 1; }
    const int ksz = gx::KSZ[kind], S = K / ksz, NT16 = N / 16, TB = 32 * bits;
    if (K % (ksz * gx::KS) || N % 128 || rpe % 16) { fprintf(stderr, "bad shape\n"); return 1; }
    const bool w16 = kind == gx::F16 || kind == gx::BF16, f4 = kind == gx::NVF4, tr3 = kind == gx::TR3FP8;
    const int ebits = w16 ? 16 : f4 ? 4 : 8, epr = 32 / ebits;
    const long R = (long)E * rpe, RB = R / 16;
    const size_t xrow = (size_t)K * ebits / 8;          // natural bytes per X row / per W column
    const float alpha = f4 ? 1.f / 256 : 1.f;
    Rng rng(a.i("seed", 1));

    // natural operands: X [R][K], W [P][N][K] (each output column contiguous along k)
    std::vector<uint8_t> tab(65536 * 2);
    for (int i = 0; i < 65536; ++i)
    {
        if (w16)
        {
            float f = rng.uni();
            if (rng.next() & 1) f = -f;
            uint16_t h = f2h(f);
            if (kind == gx::BF16) { uint32_t u; memcpy(&u, &f, 4); h = (uint16_t)(u >> 16); }
            memcpy(&tab[i * 2], &h, 2);
        }
        else if (f4) { tab[i * 2] = (uint8_t)rng.next(); tab[i * 2 + 1] = (uint8_t)rng.next(); }
        else for (int b = 0; b < 2; ++b) { const uint64_t r = rng.next(); tab[i * 2 + b] = (uint8_t)(((r >> 8) & 0x80) | ((5 + (r & 3)) << 3) | ((r >> 4) & 7)); }
    }
    auto fill = [&](std::vector<uint8_t>& v, uint64_t salt) {
        for (size_t i = 0; i + 1 < v.size(); i += 2) { const size_t h = ((i / 2 + salt) * 2654435761ull + (i >> 17)) & 65535; v[i] = tab[h * 2]; v[i + 1] = tab[h * 2 + 1]; }
    };
    std::vector<uint8_t> xn((size_t)R * xrow), wn;
    fill(xn, 0);
    if (tr3) { wn.resize((size_t)P * (K / 16) * NT16 * TB); rng.fill(wn.data(), wn.size()); }
    else { wn.resize((size_t)P * N * xrow); fill(wn, 12345); }
    std::vector<uint8_t> xsn, wsn;
    if (f4)
    {
        xsn.resize((size_t)R * (K / 16)); wsn.resize((size_t)P * N * (K / 16));
        for (auto& b : xsn) b = 0x30 + (uint8_t)(rng.next() % 24);
        for (auto& b : wsn) b = 0x30 + (uint8_t)(rng.next() % 24);
    }
    auto elem = [&](const std::vector<uint8_t>& v, size_t line, int k) -> uint32_t {
        const uint8_t* p = &v[line * xrow];
        if (w16) return (uint32_t)p[k * 2] | ((uint32_t)p[k * 2 + 1] << 8);
        if (f4) return (p[k >> 1] >> ((k & 1) * 4)) & 0xF;
        return p[k];
    };
    auto knat = [&](int s, int half, int c, int i) {
        return tr3 ? s * 32 + 16 * (i >> 1) + 2 * c + (i & 1) + 8 * half : s * ksz + half * (ksz / 2) + c * epr + i;
    };
    // fragment-ready X
    std::vector<uint32_t> xf((size_t)RB * S * 128), xsf;
    for (long rb = 0; rb < RB; ++rb)
        for (int s = 0; s < S; ++s)
            for (int l = 0; l < 32; ++l)
                for (int reg = 0; reg < 4; ++reg)
                {
                    uint32_t v = 0;
                    for (int i = 0; i < epr; ++i) v |= elem(xn, rb * 16 + (l >> 2) + (reg & 1) * 8, knat(s, reg >> 1, l & 3, i)) << (i * ebits);
                    xf[(((size_t)rb * S + s) * 32 + l) * 4 + reg] = v;
                }
    std::vector<uint32_t> wf, wsf;
    if (!tr3)
    {
        wf.resize((size_t)P * S * NT16 * 128);
        for (int p = 0; p < P; ++p)
            for (int s = 0; s < S; ++s)
                for (int tl = 0; tl < NT16; ++tl)
                    for (int l = 0; l < 32; ++l)
                        for (int reg = 0; reg < 4; ++reg)
                        {
                            uint32_t v = 0;
                            const size_t col = (size_t)p * N + tl * 16 + (reg >> 1) * 8 + (l >> 2);
                            for (int i = 0; i < epr; ++i) v |= elem(wn, col, knat(s, reg & 1, l & 3, i)) << (i * ebits);
                            wf[((((size_t)p * S + s) * NT16 + tl) * 32 + l) * 4 + reg] = v;
                        }
    }
    if (f4)
    {
        xsf.resize((size_t)RB * S * 16); wsf.resize((size_t)P * S * NT16 * 16);
        for (long rb = 0; rb < RB; ++rb)
            for (int s = 0; s < S; ++s)
                for (int r = 0; r < 16; ++r) memcpy(&xsf[((size_t)rb * S + s) * 16 + r], &xsn[(size_t)(rb * 16 + r) * (K / 16) + s * 4], 4);
        for (int p = 0; p < P; ++p)
            for (int s = 0; s < S; ++s)
                for (int tl = 0; tl < NT16; ++tl)
                    for (int cc = 0; cc < 16; ++cc)
                        memcpy(&wsf[(((size_t)p * S + s) * NT16 + tl) * 16 + cc], &wsn[((size_t)p * N + tl * 16 + cc) * (K / 16) + s * 4], 4);
    }
    std::vector<int> segs;
    for (int e = 0; e < E; ++e)
        for (int r = 0; r < rpe / 16; r += MB) { segs.push_back(e % P); segs.push_back(e * (rpe / 16) + r); segs.push_back(std::min(MB, rpe / 16 - r)); }
    const int nseg = (int)segs.size() / 3;
    const size_t wstride = tr3 ? (size_t)(K / 16) * NT16 * TB : (size_t)S * NT16 * 512, wsstride = (size_t)S * NT16 * 16;
    const uint8_t* dX = (const uint8_t*)dev_upload(xf);
    const u32* dXS = dev_upload(xsf);
    const uint8_t* dW = tr3 ? dev_upload(wn) : (const uint8_t*)dev_upload(wf);
    const u32* dWS = dev_upload(wsf);
    const int* dsegs = dev_upload(segs);
    __half* y = (__half*)dev_zero<uint16_t>((size_t)R * N);
    dim3 grid(N / 128, nseg);
    auto go = [&]() {
        #define GX_CASE(KD) case KD: if (MB == 4) gx_launch<KD, 4>(grid, dX, dXS, dW, dWS, wstride, wsstride, bits, dsegs, y, S, N, alpha); \
                                     else gx_launch<KD, 8>(grid, dX, dXS, dW, dWS, wstride, wsstride, bits, dsegs, y, S, N, alpha); break;
        switch (kind) { GX_CASE(gx::F16) GX_CASE(gx::BF16) GX_CASE(gx::FP8) GX_CASE(gx::NVF4) GX_CASE(gx::TR3FP8) }
        #undef GX_CASE
    };
    go();
    CK(cudaDeviceSynchronize());
    if (!dump.empty())
    {
        FILE* f = fopen((dump + "/meta.txt").c_str(), "w");
        fprintf(f, "kind=%s\nK=%d\nN=%d\nbits=%d\nexperts=%d\nrpe=%d\npool=%d\nMB=%d\n", kname.c_str(), K, N, bits, E, rpe, P, MB);
        fclose(f);
        dump_host(dump, "x.bin", xn.data(), xn.size());
        dump_host(dump, "w.bin", wn.data(), wn.size());
        dump_host(dump, "xs.bin", xsn.data(), xsn.size());
        dump_host(dump, "ws.bin", wsn.data(), wsn.size());
        dump_host(dump, "segs.bin", segs.data(), segs.size());
        dump_dev(dump, "y.bin", (uint16_t*)y, (size_t)R * N);
        printf("{\"cmd\":\"gemm\",\"dumped\":\"%s\"}\n", dump.c_str());
        return 0;
    }
    std::vector<double> t;
    for (int r = 0; r < reps; ++r)
    {
        const double t0 = now_us();
        go();
        CK(cudaDeviceSynchronize());
        t.push_back(now_us() - t0);
    }
    CK(cudaGetLastError());
    const Stats s = stats(t);
    const double flop = 2.0 * R * K * N;
    printf("{\"cmd\":\"gemm\",\"kind\":\"%s\",\"K\":%d,\"N\":%d,\"bits\":%d,\"MB\":%d,\"experts\":%d,\"rows_per_expert\":%d,\"rows\":%ld,\"pool\":%d,"
           "\"ms_med\":%.3f,\"ms_min\":%.3f,\"tflops_med\":%.1f,\"tflops_best\":%.1f,\"weight_mb\":%.1f}\n",
           kname.c_str(), K, N, bits, MB, E, rpe, R, P, s.med / 1e3, s.mn / 1e3, flop / s.med / 1e6, flop / s.mn / 1e6,
           (double)std::min(E, P) * (tr3 ? wstride : wstride + (f4 ? wsstride * 4 : 0)) / 1e6);
    return 0;
}
#endif
