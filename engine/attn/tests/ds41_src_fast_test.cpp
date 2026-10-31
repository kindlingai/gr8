// Host test for GR8_DS_SRC_FAST (engine/attn/ds41_src_fast.cuh + the idx wq_b GEMV route). No CUDA, no GPU.
//   g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -I engine engine/attn/tests/ds41_src_fast_test.cpp -o /tmp/ds41_src_fast_test && /tmp/ds41_src_fast_test
//
// PART B (bit-exact, bit 1): a CPU transcription of k_f32gemm2 (ds41_attn.cuh: 64 x 32 tile, 256 threads, As / Ws tiles of K steps of 32, acc[2][4] per thread,
// zero fill past M) against the very loop the new kernel runs (ds41a::src_gemv2_lane, __host__ __device__), M = 1..8, K = 5120 / 640 / 64, all fp32 bit patterns.
// What it proves: the two schedules add the products of every (row, n) in the same order from the same start, so the results are bit-identical (std::fmaf is
// the IEEE fused op on both sides). What it does not cover: the CUDA launch (smem staging, the 16 B loads), only the arithmetic loop; that needs an nvcc build and a GPU run.
//
// PART A (numerics, bit 2): the indexer q = wq_b(qr) in the two summation orders -- k_gemm_mx (per 32-block fmaf chain, acc = fmaf(part, sa * sw, acc), K order)
// against a model of the MXFP8 GEMV (activation dequantized to fp16 = e4m3 * 2^(e-127) rounded to fp16 with the 65504 clamp, per-32-group fp32 partial from 0
// of exact products, times the weight's power-of-two scale, k-part (ks) reduce). The tensor-core internal add order is not bit-emulated (the group partial is
// summed in double and rounded to fp32: a better-than-hardware reference). Then logits_j = sum_h w_h relu(q_h . k_j) over random keys and the top-512 set of
// each. It reports how many q elements move by one bf16 step and whether the top-512 sets are equal; sets can differ only at the 512th-place near-tie.
#include "../ds41_src_fast.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using ds41a::SfV4;
static uint16_t f2bf(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static float bf2f(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static uint32_t bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// ---------------------------------------------------------------------------------------------- PART B
// k_f32gemm2, thread by thread (blockIdx.z picks the weight; the test runs one weight, the other is the same code)
static void ref_f32gemm2(const std::vector<uint16_t>& x, const std::vector<uint16_t>& W, std::vector<float>& Y, int M, int N, int K)
{
    constexpr int FB_M = 32, FB_N = 64;
    for (int by = 0; by < (M + FB_M - 1) / FB_M; by++)
        for (int bx = 0; bx < N / FB_N; bx++)
        {
            float As[FB_M][33], Ws[FB_N][33];
            float acc[256][2][4] = {};
            const int m0 = by * FB_M, n0 = bx * FB_N;
            for (int k0 = 0; k0 < K; k0 += 32)
            {
                for (int i = 0; i < FB_M * 32; i++) As[i >> 5][i & 31] = (m0 + (i >> 5) < M) ? bf2f(x[(size_t)(m0 + (i >> 5)) * K + k0 + (i & 31)]) : 0.0f;
                for (int i = 0; i < FB_N * 32; i++) Ws[i >> 5][i & 31] = (n0 + (i >> 5) < N) ? bf2f(W[(size_t)(n0 + (i >> 5)) * K + k0 + (i & 31)]) : 0.0f;
                for (int tid = 0; tid < 256; tid++)
                {
                    const int ty = tid >> 4, tx = tid & 15;
                    for (int k = 0; k < 32; k++)
                    {
                        const float a0 = As[ty * 2][k], a1 = As[ty * 2 + 1][k];
                        float w[4];
                        for (int j = 0; j < 4; j++) w[j] = Ws[tx * 4 + j][k];
                        for (int j = 0; j < 4; j++)
                        {
                            acc[tid][0][j] = std::fmaf(a0, w[j], acc[tid][0][j]);
                            acc[tid][1][j] = std::fmaf(a1, w[j], acc[tid][1][j]);
                        }
                    }
                }
            }
            for (int tid = 0; tid < 256; tid++)
            {
                const int ty = tid >> 4, tx = tid & 15;
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 4; j++)
                        if (m0 + ty * 2 + i < M && n0 + tx * 4 + j < N) Y[(size_t)(m0 + ty * 2 + i) * N + n0 + tx * 4 + j] = acc[tid][i][j];
            }
        }
}
// k_f32gemv2: block (bx), warp w, lane
static void new_f32gemv2(const std::vector<uint16_t>& x, const std::vector<uint16_t>& W, std::vector<float>& Y, int M, int N, int K)
{
    std::vector<SfV4> xs((size_t)M * (K / 8)), wv((size_t)N * (K / 8));
    std::memcpy(xs.data(), x.data(), (size_t)M * K * 2);
    std::memcpy(wv.data(), W.data(), (size_t)N * K * 2);
    const int warps = (M + ds41a::SF_R - 1) / ds41a::SF_R;
    for (int bx = 0; bx < N / 32; bx++)
        for (int w = 0; w < warps; w++)
            for (int lane = 0; lane < 32; lane++)
            {
                const int n = bx * 32 + lane, r0 = w * ds41a::SF_R;
                float acc[ds41a::SF_R] = {};
                ds41a::src_gemv2_lane<ds41a::SF_R>(wv.data() + (size_t)n * (K / 8), xs.data(), K / 8, r0, M, K / 64, acc);
                for (int r = 0; r < ds41a::SF_R; r++)
                    if (r0 + r < M) Y[(size_t)(r0 + r) * N + n] = acc[r];
            }
}
static uint16_t rnd_bf(std::mt19937& g, int elo, int ehi)
{
    const uint32_t r = g();
    if ((r & 63) == 0) return (r & 64) ? 0x8000 : 0;   // +-0
    const int e = elo + (int)((r >> 8) % (uint32_t)(ehi - elo + 1));
    const float m = 1.0f + (float)((g() & 0xffff)) / 65536.0f;
    float f = std::ldexp(m, e);
    if (r & 128) f = -f;
    return f2bf(f);
}
static void part_b()
{
    std::mt19937 g(12345);
    const int Ks[] = {64, 640, 5120}, N = 64;
    const int ranges[][2] = {{-18, 8}, {-60, 60}};   // typical, and a wide one that reaches fp32 denormals / overflow in the products
    long cmp = 0, bad = 0;
    for (int K : Ks)
        for (auto& rg : ranges)
            for (int M = 1; M <= 8; M++)
            {
                std::vector<uint16_t> x((size_t)M * K), W((size_t)N * K);
                for (auto& v : x) v = rnd_bf(g, rg[0], rg[1]);
                for (auto& v : W) v = rnd_bf(g, rg[0], rg[1]);
                std::vector<float> Yr((size_t)M * N, -1.f), Yn((size_t)M * N, -2.f);
                ref_f32gemm2(x, W, Yr, M, N, K);
                new_f32gemv2(x, W, Yn, M, N, K);
                long b = 0;
                for (size_t i = 0; i < Yr.size(); i++)
                {
                    cmp++;
                    if (bits(Yr[i]) != bits(Yn[i])) b++;
                }
                bad += b;
                CHECK(b == 0, "PART B K=%d range=%d..%d M=%d: %ld of %zu outputs differ", K, rg[0], rg[1], M, b, Yr.size());
            }
    std::printf("PART B: %ld outputs compared (K 64/640/5120, M 1..8, two magnitude ranges), %ld bit differences\n", cmp, bad);
}

// ---------------------------------------------------------------------------------------------- PART A
static float e4m3(uint8_t c)
{
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
    return s ? -v : v;
}
static float e8m0(uint8_t b) { return std::ldexp(1.0f, (int)b - 127); }
static float to_f16(float v)   // round to nearest fp16 value, clamp +-65504 (the xfrag kernel's fminf/fmaxf then pack_h2)
{
    if (v > 65504.f) return 65504.f;
    if (v < -65504.f) return -65504.f;
    if (std::fabs(v) < 6.103515625e-05f) return std::nearbyintf(v * 16777216.0f) / 16777216.0f;   // fp16 subnormal grid 2^-24
    return v;   // e4m3 * 2^e has 4 significant bits: exact in fp16's 11
}
static uint8_t rnd_code(std::mt19937& g)
{
    for (;;)
    {
        const uint8_t c = (uint8_t)g();
        if ((c & 0x7f) == 0x7f) continue;   // e4m3fn NaN
        // bias toward moderate magnitudes: re-draw a third of the time if the exponent field is top
        if (((c >> 3) & 15) >= 13 && (g() % 3)) continue;
        return c;
    }
}
static void part_a(int rows)
{
    const int N = 4096, K = 1280, KB = K / 32, NK = 16384, NH = 32, HD = 128, TOPK = 512;
    std::mt19937 g(777 + rows);
    std::vector<uint8_t> Wc((size_t)N * K), Ws((size_t)N * KB), Ac((size_t)rows * K), As((size_t)rows * KB);
    for (auto& c : Wc) c = rnd_code(g);
    for (auto& c : Ac) c = rnd_code(g);
    for (auto& s : Ws) s = (uint8_t)(112 + g() % 6);   // 2^-15 .. 2^-10
    for (auto& s : As) s = (uint8_t)(116 + g() % 6);
    std::vector<float> keys((size_t)NK * HD), wh((size_t)rows * NH);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : keys) v = nd(g);
    for (auto& v : wh) v = std::fabs(nd(g));
    // the two q's
    std::vector<uint16_t> qo((size_t)rows * N), qg((size_t)rows * N);
    long moved = 0, moved_max_ulp = 0;
    double ss = 0, maxabs = 0;
    const int ks = 8;   // gv_pick_ks(1280): 40 records -> 8 k-parts of 5
    for (int m = 0; m < rows; m++)
        for (int n = 0; n < N; n++)
        {
            float acc = 0.0f;   // k_gemm_mx
            for (int kb = 0; kb < KB; kb++)
            {
                float part = 0.0f;
                for (int k = 0; k < 32; k++) part = std::fmaf(e4m3(Ac[(size_t)m * K + kb * 32 + k]), e4m3(Wc[(size_t)n * K + kb * 32 + k]), part);
                acc = std::fmaf(part, e8m0(As[(size_t)m * KB + kb]) * e8m0(Ws[(size_t)n * KB + kb]), acc);
            }
            qo[(size_t)m * N + n] = f2bf(acc);
            float parts[16] = {};   // GEMV model
            for (int kb = 0; kb < KB; kb++)
            {
                double gsum = 0;
                const float sa = e8m0(As[(size_t)m * KB + kb]);
                for (int k = 0; k < 32; k++)
                    gsum += (double)to_f16(e4m3(Ac[(size_t)m * K + kb * 32 + k]) * sa) * (double)e4m3(Wc[(size_t)n * K + kb * 32 + k]);
                parts[kb / (KB / ks)] = parts[kb / (KB / ks)] + (float)gsum * e8m0(Ws[(size_t)n * KB + kb]);
            }
            float tot = 0.0f;
            for (int p = 0; p < ks; p++) tot += parts[p];
            qg[(size_t)m * N + n] = f2bf(tot);
            ss += (double)bf2f(qo[(size_t)m * N + n]) * bf2f(qo[(size_t)m * N + n]);
            maxabs = std::max(maxabs, (double)std::fabs(bf2f(qo[(size_t)m * N + n]) - bf2f(qg[(size_t)m * N + n])));
            if (qo[(size_t)m * N + n] != qg[(size_t)m * N + n])
            {
                moved++;
                const long d = std::labs((long)qo[(size_t)m * N + n] - (long)qg[(size_t)m * N + n]);
                if (d > moved_max_ulp) moved_max_ulp = d;
            }
        }
    // logits and top-512 per row
    int set_equal = 0;
    long set_diff_total = 0;
    double worst_gap = 0;
    for (int m = 0; m < rows; m++)
    {
        std::vector<float> lo(NK), lg(NK);
        for (int j = 0; j < NK; j++)
        {
            float so = 0, sg = 0;
            for (int h = 0; h < NH; h++)
            {
                float dO = 0, dG = 0;
                for (int d = 0; d < HD; d++)
                {
                    dO += bf2f(qo[(size_t)m * N + h * HD + d]) * keys[(size_t)j * HD + d];
                    dG += bf2f(qg[(size_t)m * N + h * HD + d]) * keys[(size_t)j * HD + d];
                }
                so += wh[(size_t)m * NH + h] * std::max(dO, 0.f);
                sg += wh[(size_t)m * NH + h] * std::max(dG, 0.f);
            }
            lo[j] = so; lg[j] = sg;
        }
        auto topset = [&](const std::vector<float>& l, float& cut) {
            std::vector<int> id(NK);
            for (int j = 0; j < NK; j++) id[j] = j;
            std::partial_sort(id.begin(), id.begin() + TOPK + 1, id.end(), [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
            cut = l[id[TOPK - 1]];
            std::vector<int> s(id.begin(), id.begin() + TOPK);
            std::sort(s.begin(), s.end());
            return s;
        };
        float co, cg;
        const std::vector<int> so = topset(lo, co), sg = topset(lg, cg);
        std::vector<int> d;
        std::set_symmetric_difference(so.begin(), so.end(), sg.begin(), sg.end(), std::back_inserter(d));
        if (d.empty()) set_equal++;
        set_diff_total += (long)d.size() / 2;
        for (int j : d) worst_gap = std::max(worst_gap, std::fabs((double)lo[j] - co) / std::fabs((double)co));
    }
    std::printf("PART A rows=%d: max |q_old - q_gv| = %.2e x rms(q); q elements %ld of %zu differ by a bf16 step (max %ld code steps, near zero); top-%d sets equal in %d/%d rows, %ld swapped members in total, worst |logit - cutoff| / cutoff of a swapped member = %.2e\n",
                rows, maxabs / std::sqrt(ss / (double)qo.size()), moved, qo.size(), moved_max_ulp, TOPK, set_equal, rows, set_diff_total, worst_gap);
}

int main()
{
    part_b();
    part_a(6);
    part_a(8);
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
