// Bit-equality gate of k_gate_logits_ro (GR8_DS_GATE_RO=1; ds_router.cuh, ds41_moe_ops.cuh router_ro) against k_gate_logits_mt.
//   (no argument)  host mode. Runs on any machine and makes no CUDA call. It runs gro_group / gro_rows (the same source the kernel runs, compiled for the CPU) lane by lane
//                  for every CTA slice and warp, then the xor butterfly and the store guard as the kernel writes them, and compares every logit with an independent
//                  reference written from k_gate_logits_mt (per lane: acc = 0.f, i = 0..19, j = 0..7, fmaf(w, x, acc); then butterfly 16, 8, 4, 2, 1). Bytes must be equal for
//                  every (token, expert); every output cell must be written exactly once. A mutant reference (j order reversed) must differ (the comparator can fail).
//                  This is the CPU run of the kernel's arithmetic and tile maps; it is not the compiled kernel (no SASS run).
//   gpu            GPU mode (a spark; refuses anything but sm_121). Both kernels run on the same random bf16 data for M in {1, 7, 8, 9, 33, 255, 256, 257, 846, 1000, 2047, 2048}, 384
//                  experts; the whole logits buffers (also the rows >= M, pre-filled with 0xFF bytes) are compared with memcmp. Then the two kernels are timed at M 2048 and 846.
//                  Exit 0 = every line identical=yes.
// Build (compiles on any CUDA host; gpu mode needs sm_121): nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -Xcompiler -Wall -I engine/kernels -o ds_gate_ro_test engine/kernels/tests/ds_gate_ro_test.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../ds_router.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

using namespace dsr;

static float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t to_bf16(float f)       // round to nearest even, finite inputs
{
    uint32_t u; std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// k_gate_logits_mt's order, written from scratch: one logit.
static float ref_logit(const uint16_t* w, const uint16_t* x, bool rev_j, bool* lanes_agree)
{
    float acc[32];
    for (int lane = 0; lane < 32; ++lane)
    {
        float a = 0.f;
        for (int i = 0; i < GRO_NV; ++i)
            for (int j = 0; j < 8; ++j)
            {
                const int k = 8 * (lane + 32 * i) + (rev_j ? 7 - j : j);
                a = fmaf(bf(w[k]), bf(x[k]), a);
            }
        acc[lane] = a;
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        float nx[32];
        for (int l = 0; l < 32; ++l) nx[l] = acc[l] + acc[l ^ o];
        std::memcpy(acc, nx, sizeof acc);
    }
    if (lanes_agree)
        for (int l = 1; l < 32; ++l)
            if (std::memcmp(&acc[l], &acc[0], 4) != 0) *lanes_agree = false;
    return acc[0];
}

struct Data
{
    std::vector<uint4> gq, xq;             // 16-byte aligned views of the bf16 arrays
    uint16_t* g16; uint16_t* x16;
    int NT, M;
};
static void make_data(Data& d, int NT, int M, int mode, unsigned seed)
{
    d.NT = NT; d.M = M;
    d.gq.assign((size_t)NT * GRO_PIECES, uint4{0, 0, 0, 0});
    d.xq.assign((size_t)M * GRO_PIECES, uint4{0, 0, 0, 0});
    d.g16 = reinterpret_cast<uint16_t*>(d.gq.data());
    d.x16 = reinterpret_cast<uint16_t*>(d.xq.data());
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ex(-8, 8);
    auto fill = [&](uint16_t* p, size_t n, float sc)
    {
        for (size_t k = 0; k < n; ++k)
        {
            float v = nd(rng) * sc;
            if (mode == 0) v = std::ldexp(v, ex(rng));          // exponent spread: the chain's rounding depends on the order
            if (rng() % 97 == 0) v = 0.f;
            p[k] = to_bf16(v);
        }
    };
    fill(d.g16, (size_t)NT * HIDDEN, mode == 0 ? 1.f : 0.05f);
    fill(d.x16, (size_t)M * HIDDEN, 1.f);
}

// the kernel body for one CTA slice (b), warp by warp, lane by lane; the butterfly and the store guard as in k_gate_logits_ro
static void emulate_ro(const Data& d, std::vector<float>& out, std::vector<int>& writes)
{
    const int NT = d.NT, M = d.M;
    std::vector<uint4> sw((size_t)GRO_EXP * GRO_PIECES);
    for (int b = 0; b < NT / GRO_EXP; ++b)
    {
        const int e0 = b * GRO_EXP;
        std::memcpy(sw.data(), d.gq.data() + (size_t)e0 * GRO_PIECES, sw.size() * sizeof(uint4));
        for (int wid = 0; wid < 8; ++wid)
            for (int g = wid; g * GRO_TOK < M; g += 8)
            {
                int ro[GRO_TOK];
                gro_rows(g * GRO_TOK, M, ro);
                static float lane_acc[32][GRO_EXP][GRO_TOK];
                for (int lane = 0; lane < 32; ++lane)
                    gro_group(d.xq.data() + lane, ro, sw.data() + lane, lane_acc[lane]);
                for (int e = 0; e < GRO_EXP; ++e)
                    for (int t = 0; t < GRO_TOK; ++t)
                    {
                        float a[32];
                        for (int l = 0; l < 32; ++l) a[l] = lane_acc[l][e][t];
                        for (int o = 16; o > 0; o >>= 1)
                        {
                            float nx[32];
                            for (int l = 0; l < 32; ++l) nx[l] = a[l] + a[l ^ o];     // lane l: acc += shfl_xor(acc, o)
                            std::memcpy(a, nx, sizeof a);
                        }
                        if (g * GRO_TOK + t < M)                                      // lane 0 stores
                        {
                            const size_t at = (size_t)(g * GRO_TOK + t) * NT + e0 + e;
                            out[at] = a[0];
                            ++writes[at];
                        }
                    }
            }
    }
}

static int run_host()
{
    struct Shape { int NT, M, mode; };
    const Shape shapes[] = { {8, 1, 0}, {8, 7, 0}, {8, 8, 0}, {16, 9, 0}, {16, 33, 0}, {24, 255, 0}, {24, 257, 1}, {16, 846, 0}, {8, 1000, 1}, {8, 2047, 0}, {8, 2048, 1}, {24, 300, 0} };
    int bad_total = 0;
    for (const Shape& sh : shapes)
    {
        Data d; make_data(d, sh.NT, sh.M, sh.mode, 1234u + (unsigned)sh.M);
        std::vector<float> out((size_t)sh.M * sh.NT, 0.f);
        std::vector<int> writes(out.size(), 0);
        emulate_ro(d, out, writes);
        long bad = 0, bad_mut = 0, nowr = 0; bool agree = true;
        for (int m = 0; m < sh.M; ++m)
            for (int e = 0; e < sh.NT; ++e)
            {
                const size_t at = (size_t)m * sh.NT + e;
                const float r = ref_logit(d.g16 + (size_t)e * HIDDEN, d.x16 + (size_t)m * HIDDEN, false, &agree);
                if (std::memcmp(&r, &out[at], 4) != 0) ++bad;
                if (writes[at] != 1) ++nowr;
                if (sh.M <= 33)
                {
                    const float q = ref_logit(d.g16 + (size_t)e * HIDDEN, d.x16 + (size_t)m * HIDDEN, true, nullptr);
                    if (std::memcmp(&q, &out[at], 4) != 0) ++bad_mut;
                }
            }
        const bool ok = bad == 0 && nowr == 0 && agree;
        printf("host NT %2d M %4d data %d: logits %zu bad=%ld not_written_once=%ld lanes_agree=%s%s", sh.NT, sh.M, sh.mode, out.size(), bad, nowr, agree ? "yes" : "NO", ok ? "" : "  <-- FAIL");
        if (sh.M <= 33) printf("  | mutant(j reversed) differs in %ld of %zu (expected > 0)", bad_mut, out.size());
        printf("\n");
        if (!ok) ++bad_total;
        if (sh.M <= 33 && bad_mut == 0) { printf("  mutant did not differ: the comparator has no power on this shape  <-- FAIL\n"); ++bad_total; }
    }
    printf("ds_gate_ro host: %s\n", bad_total == 0 ? "ALL IDENTICAL" : "FAIL");
    return bad_total ? 1 : 0;
}

static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

static int run_gpu()
{
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    if (pr.major != 12 || pr.minor != 1) { printf("refusing to run: sm_%d%d is not sm_121\n", pr.major, pr.minor); return 3; }
    constexpr int NE = NEXP, MAXM = 2048;
    CK(cudaFuncSetAttribute(k_gate_logits_ro<NEXP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GRO_SMEM));
    const int Ms[] = { 1, 7, 8, 9, 33, 255, 256, 257, 846, 1000, 2047, 2048 };
    int bad_total = 0;
    __nv_bfloat16 *dg = nullptr, *dx = nullptr; float *lo = nullptr, *ln = nullptr;
    CK(cudaMalloc(&dg, (size_t)NE * HIDDEN * 2)); CK(cudaMalloc(&dx, (size_t)MAXM * HIDDEN * 2));
    CK(cudaMalloc(&lo, (size_t)MAXM * NE * 4)); CK(cudaMalloc(&ln, (size_t)MAXM * NE * 4));
    std::vector<float> ho((size_t)MAXM * NE), hn((size_t)MAXM * NE);
    for (int mode = 0; mode < 2; ++mode)
    {
        Data d; make_data(d, NE, MAXM, mode, 777u + mode);
        CK(cudaMemcpy(dg, d.g16, (size_t)NE * HIDDEN * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, d.x16, (size_t)MAXM * HIDDEN * 2, cudaMemcpyHostToDevice));
        for (int M : Ms)
        {
            CK(cudaMemset(lo, 0xFF, (size_t)MAXM * NE * 4)); CK(cudaMemset(ln, 0xFF, (size_t)MAXM * NE * 4));
            k_gate_logits_mt<NE, 0><<<NE / 8, 256>>>(dg, dx, M, lo);
            k_gate_logits_ro<NE><<<NE / 8, 256, GRO_SMEM>>>(dg, dx, M, ln);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(ho.data(), lo, ho.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hn.data(), ln, hn.size() * 4, cudaMemcpyDeviceToHost));
            long diff = 0;
            for (size_t k = 0; k < ho.size(); ++k) if (std::memcmp(&ho[k], &hn[k], 4) != 0) ++diff;
            printf("gpu data %d M %4d: identical=%s differing_words=%ld (whole %d x %d buffer incl. rows >= M)\n", mode, M, diff == 0 ? "yes" : "NO", diff, MAXM, NE);
            if (diff) ++bad_total;
        }
    }
    for (int M : { 2048, 846 })
    {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        auto time_one = [&](bool ro)
        {
            std::vector<float> ms;
            for (int it = 0; it < 25; ++it)
            {
                CK(cudaEventRecord(a));
                if (ro) k_gate_logits_ro<NE><<<NE / 8, 256, GRO_SMEM>>>(dg, dx, M, ln);
                else k_gate_logits_mt<NE, 0><<<NE / 8, 256>>>(dg, dx, M, lo);
                CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
                float t; CK(cudaEventElapsedTime(&t, a, b)); if (it >= 5) ms.push_back(t);
            }
            return median(ms);
        };
        const float t_old = time_one(false), t_new = time_one(true);
        printf("gpu time M %4d (median of 20, back-to-back, hot caches): k_gate_logits_mt %.3f ms, k_gate_logits_ro %.3f ms\n", M, t_old, t_new);
    }
    printf("ds_gate_ro gpu: %s\n", bad_total == 0 ? "ALL IDENTICAL" : "FAIL");
    return bad_total ? 1 : 0;
}

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "gpu") return run_gpu();
    return run_host();
}
