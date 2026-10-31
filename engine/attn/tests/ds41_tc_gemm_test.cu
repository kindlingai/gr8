// Tensor-core block-scaled fp8 GEMM (k_gemm_mx8t) vs the scalar k_gemm_mxfp8 vs a double host reference, on the five attention projection shapes plus the indexer's wq_b shape.
// Synthetic MXFP8 operands (valid e4m3 codes, E8M0 bytes 125..129). usage: ds41_tc_gemm_test [--M 16,100,256,512,1024,2048] [--reps 20] [--shapes wq_a,wkv,wq_b,wo_a,wo_b,idx_wqb,engram_wkv (default: all but engram_wkv)] [--sweep: time each of the 5 tiles] [--ceiling: mma-only loop, 1..3 CTAs/SM] [--mfauto: host only, print the GR8_DS_ATTN_TC_MF=2 launch-order pick per shape and M, exit 1 if it is not M-fastest for exactly the Engram wkv at 2+ row tiles] [--mfast: also time the default launch order against the M-tile-fastest order (g_tc_mfast) and compare the two outputs bit for bit]
// Prints per shape and M: rel-L2 of scalar / tc vs the double reference on sampled outputs, rel-L2 tc vs scalar (full output), us and TFLOP/s of both.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#define G6_MODEL_DS41 1
#include "../ds41_attn.cu"
using namespace ds41a;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
static double e4m3(uint8_t c)
{
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    const double v = e == 0 ? (m / 8.0) * std::ldexp(1.0, -6) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return s ? -v : v;
}
struct Shape { const char* nm; int N, K, G; };
// mma-only ceiling: every warp issues 16 independent block-scaled mma (m16n8k32) per iteration on fixed registers, no loads, no stores of note
__global__ void __launch_bounds__(256) k_mma_ceiling(float* out, int iters)
{
    float acc[4][4][4];
    for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) for (int z = 0; z < 4; z++) acc[x][y][z] = 0.f;
    const uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};   // e4m3 1.0
    for (int i = 0; i < iters; i++)
#pragma unroll
        for (int x = 0; x < 4; x++)
#pragma unroll
            for (int y = 0; y < 4; y++) tc_mma(acc[x][y], a, 0x38383838u, 0x38383838u, 127u, 127u);
    float s = 0.f;
    for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) for (int z = 0; z < 4; z++) s += acc[x][y][z];
    if (s == 1.2345f) out[0] = s;
}
int main(int argc, char** argv)
{
    std::vector<int> Ms = {16, 100, 256, 512, 1024, 2048};
    int reps = 20;
    std::string shapes;
    bool sweep = false, ceiling = false, mfast = false, mfauto = false;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--M") && i + 1 < argc) { Ms.clear(); for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) Ms.push_back(atoi(t)); }
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) shapes = std::string(",") + argv[++i] + ",";
        else if (!strcmp(argv[i], "--sweep")) sweep = true;
        else if (!strcmp(argv[i], "--ceiling")) ceiling = true;
        else if (!strcmp(argv[i], "--mfast")) mfast = true;
        else if (!strcmp(argv[i], "--mfauto")) mfauto = true;
    }
    const Shape sh[] = {{"wq_a", 1280, 5120, 1}, {"wkv", 512, 5120, 1}, {"wq_b", 8192, 1280, 1}, {"wo_a", 1024, 4096, 2}, {"wo_b", 5120, 2048, 1}, {"idx_wqb", 4096, 1280, 1}, {"engram_wkv", 25600, 6144, 1}};   // idx_wqb = the indexer's wq_b (N 4096, K 1280)
    if (mfauto)
    {   // host only, no GPU: the GR8_DS_ATTN_TC_MF=2 pick per shape and M (the tile row BM = 128 / 64 / 32 as launch_gemm_tc chooses it). Expect: M-fastest only for engram_wkv with M > 128 (2+ row tiles); mode 0 never; mode 1 always.
        int mbad = 0;
        printf("# GR8_DS_ATTN_TC_MF=2 auto pick: weight bytes N*K*G vs TC_MF_AUTO_BYTES %zu\n", TC_MF_AUTO_BYTES);
        for (const Shape& S : sh)
        {
            const size_t wb = (size_t)S.N * S.K * S.G;
            printf("%-10s %9.1f MB |", S.nm, wb / 1e6);
            for (int M : Ms)
            {
                const int BM = M >= 128 ? 128 : (M >= 64 ? 64 : 32);
                const unsigned mt = (M + BM - 1) / BM;
                const bool a = tc_mfast_pick(2, S.N, S.K, S.G, mt), z = tc_mfast_pick(0, S.N, S.K, S.G, mt), o = tc_mfast_pick(1, S.N, S.K, S.G, mt);
                const bool want = !strcmp(S.nm, "engram_wkv") && mt >= 2;
                if (a != want || z || !o) { mbad++; printf(" M%d WRONG(auto %d mode0 %d mode1 %d)", M, a, z, o); } else printf(" M%d %s", M, a ? "M-fastest" : "N-fastest");
            }
            printf("\n");
        }
        printf("%s\n", mbad ? "MFAUTO_BAD" : "MFAUTO_OK");
        return mbad ? 1 : 0;
    }
    std::mt19937 rng(1234);
    auto rcode = [&]() { uint8_t c; do { c = (uint8_t)(rng() & 0xFF); } while ((c & 0x7F) == 0x7F); return c; };
    int bad = 0;
    {   // ds41_gemm_mx_tc rejects without launching (the pointers are never read on these paths): 1 = shape, 2 = alignment; good shape + aligned = 0 would launch, so not tried here
        const uint8_t* q = (const uint8_t*)0x10000;
        bf16* y = (bf16*)0x20000;
        const int r[5] = {ds41_gemm_mx_tc(q, 1280, q, 40, q, q, y, 4096, 64, 4096, 1000, nullptr),    // K % 128
                          ds41_gemm_mx_tc(q, 1280, q, 40, q, q, y, 4096, 64, 4090, 1280, nullptr),    // N % 64
                          ds41_gemm_mx_tc(q, 1280, q, 40, q, q, y, 4096, 0, 4096, 1280, nullptr),     // M >= 1
                          ds41_gemm_mx_tc(q, 1280, q, 40, q, q, y, 4095, 64, 4096, 1280, nullptr),    // ldy even
                          ds41_gemm_mx_tc(q + 1, 1280, q, 40, q, q, y, 4096, 64, 4096, 1280, nullptr)};   // A 16 B aligned
        const int want[5] = {1, 1, 1, 1, 2};
        for (int i = 0; i < 5; i++) if (r[i] != want[i]) { printf("ENTRY_REJECT_BAD case %d rc %d want %d\n", i, r[i], want[i]); bad++; }
        CK(cudaGetLastError());
    }
    printf("# device ");
    { cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); printf("%s sm_%d%d\n", pr.name, pr.major, pr.minor); }
    printf("# shape M | rel-L2 vs double ref (sampled): scalar tc | rel-L2 tc vs scalar (full) | scalar us tc us | scalar TF/s tc TF/s | speedup\n");
    if (ceiling)
    {
        int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
        float* d; CK(cudaMalloc(&d, 4));
        const int iters = 50000;
        printf("# mma-only ceiling (m16n8k32 e4m3 block-scaled, no loads), %d SMs: CTAs/SM x 8 warps\n", nsm);
        for (int cps = 1; cps <= 3; cps++)
        {
            k_mma_ceiling<<<nsm * cps, 256>>>(d, 100); CK(cudaDeviceSynchronize());
            cudaEvent_t c0, c1; cudaEventCreate(&c0); cudaEventCreate(&c1);
            cudaEventRecord(c0); k_mma_ceiling<<<nsm * cps, 256>>>(d, iters); cudaEventRecord(c1); CK(cudaEventSynchronize(c1));
            float ms; cudaEventElapsedTime(&ms, c0, c1);
            const double fl = (double)nsm * cps * 8 * iters * 16 * 8192.0;
            printf("ceiling %d CTA/SM (%d warps/SM): %.1f us  %.1f TF/s\n", cps, cps * 8, ms * 1000.0, fl / (ms * 1e-3) / 1e12);
        }
        cudaFree(d);
    }
    for (const Shape& S : sh)
        for (int M : Ms)
        {
            if (shapes.empty() ? !strcmp(S.nm, "engram_wkv") : shapes.find(std::string(",") + S.nm + ",") == std::string::npos) continue;
            const int N = S.N, K = S.K, G = S.G, KB = K / 32;
            std::vector<uint8_t> hA((size_t)M * K * G), hsA((size_t)M * KB * G), hW((size_t)N * K * G), hsW((size_t)N * KB * G);
            for (auto& v : hA) v = rcode();
            for (auto& v : hW) v = rcode();
            for (auto& v : hsA) v = (uint8_t)(125 + rng() % 5);
            for (auto& v : hsW) v = (uint8_t)(125 + rng() % 5);
            uint8_t *dA, *dsA, *dW, *dsW;
            bf16 *dYs, *dYt;
            CK(cudaMalloc(&dA, hA.size())); CK(cudaMalloc(&dsA, hsA.size())); CK(cudaMalloc(&dW, hW.size())); CK(cudaMalloc(&dsW, hsW.size()));
            CK(cudaMalloc(&dYs, (size_t)M * N * G * 2)); CK(cudaMalloc(&dYt, (size_t)M * N * G * 2));
            CK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsA, hsA.data(), hsA.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsW, hsW.data(), hsW.size(), cudaMemcpyHostToDevice));
            // group layout as the engine's: A rows of one group are K bytes apart with group stride K (lda = K * G for a G-group o_a input), Y columns G * N
            const int lda = K * G, lsa = KB * G, ldy = N * G;
            Fp8Mat W{dW, dsW, N, K, nullptr};
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            float us[2] = {0, 0};
            for (int tc = 0; tc < 2; tc++)
            {
                g_use_tc = tc != 0;
                bf16* Y = tc ? dYt : dYs;
                launch_gemm(dA, lda, dsA, lsa, W, Y, ldy, M, G, K, KB, N, 0);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                cudaEventRecord(e0);
                for (int r = 0; r < reps; r++) launch_gemm(dA, lda, dsA, lsa, W, Y, ldy, M, G, K, KB, N, 0);
                cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                us[tc] = ms * 1000.0f / reps;
            }
            g_use_tc = false;
            std::vector<uint16_t> ys((size_t)M * N * G), yt((size_t)M * N * G);
            CK(cudaMemcpy(ys.data(), dYs, ys.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(yt.data(), dYt, yt.size() * 2, cudaMemcpyDeviceToHost));
            long ediff = -1; int erc = -1;   // the C entry (G == 1 shapes) must give the same bits as launch_gemm with g_use_tc
            if (G == 1)
            {
                bf16* dYe; CK(cudaMalloc(&dYe, (size_t)M * N * 2)); CK(cudaMemset(dYe, 0, (size_t)M * N * 2));
                erc = ds41_gemm_mx_tc(dA, lda, dsA, lsa, dW, dsW, dYe, ldy, M, N, K, nullptr);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                std::vector<uint16_t> ye((size_t)M * N);
                CK(cudaMemcpy(ye.data(), dYe, ye.size() * 2, cudaMemcpyDeviceToHost));
                ediff = 0; for (size_t i = 0; i < ye.size(); i++) ediff += ye[i] != yt[i];
                cudaFree(dYe);
            }
            auto bf = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return (double)f; };
            double dn = 0, dd = 0;
            for (size_t i = 0; i < ys.size(); i++) { const double a = bf(ys[i]), b = bf(yt[i]); dn += (a - b) * (a - b); dd += a * a; }
            double en[2] = {0, 0}, ed = 0;
            for (int smp = 0; smp < 96; smp++)
            {
                const int g = rng() % G, m = rng() % M, n = rng() % N;
                double ref = 0;
                for (int kb = 0; kb < KB; kb++)
                {
                    double dot = 0;
                    for (int k = 0; k < 32; k++)
                        dot += e4m3(hA[(size_t)m * lda + (size_t)g * K + kb * 32 + k]) * e4m3(hW[((size_t)g * N + n) * K + kb * 32 + k]);
                    ref += dot * std::ldexp(1.0, hsA[(size_t)m * lsa + (size_t)g * KB + kb] - 127) * std::ldexp(1.0, hsW[((size_t)g * N + n) * KB + kb] - 127);
                }
                const size_t o = (size_t)m * ldy + (size_t)g * N + n;
                const double a = bf(ys[o]), b = bf(yt[o]);
                en[0] += (a - ref) * (a - ref); en[1] += (b - ref) * (b - ref); ed += ref * ref;
            }
            const double fl = 2.0 * M * N * K * G;
            const bool ok = std::sqrt(en[1] / ed) < 0.01 && std::sqrt(dn / dd) < 0.01 && (G != 1 || (erc == 0 && ediff == 0));
            if (!ok) bad++;
            printf("%-5s M %4d | %.2e %.2e | %.2e | %8.1f %8.1f | %6.2f %7.2f | x%.1f entry rc %d diff %ld %s\n", S.nm, M, std::sqrt(en[0] / ed), std::sqrt(en[1] / ed), std::sqrt(dn / dd), us[0], us[1],
                   fl / (us[0] * 1e6), fl / (us[1] * 1e6), us[0] / us[1], erc, ediff, ok ? "" : "BAD");
            if (sweep && G == 1)
            {   // each tile forced through launch_tc_t (N % BN == 0 only); the default pick is the "tc" column above
                GemmP gp; gp.A = dA; gp.SA = dsA; gp.W = dW; gp.SW = dsW; gp.Y = dYt; gp.M = M; gp.N = N; gp.K = K; gp.lda = lda; gp.lsa = lsa; gp.ldy = ldy;
                gp.a_gs = gp.sa_gs = gp.y_gs = 0; gp.w_gs = (size_t)N * K; gp.sw_gs = (size_t)N * KB;
                printf("sweep %-10s M %4d |", S.nm, M);
#define SWEEP_TILE(BM_, BN_, ST_)                                                                                                         \
                if (N % BN_ == 0)                                                                                                             \
                {                                                                                                                             \
                    launch_tc_t<BM_, BN_, ST_>(gp, 1, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());                                \
                    cudaEventRecord(e0);                                                                                                      \
                    for (int r = 0; r < reps; r++) launch_tc_t<BM_, BN_, ST_>(gp, 1, 0);                                                      \
                    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));                                                                        \
                    float ms_; cudaEventElapsedTime(&ms_, e0, e1);                                                                            \
                    printf(" %dx%dx%d %.1f us %.1f TF/s |", BM_, BN_, ST_, ms_ * 1000.0f / reps, fl / (ms_ * 1000.0f / reps * 1e6));        \
                }
                SWEEP_TILE(32, 64, 3) SWEEP_TILE(64, 64, 3) SWEEP_TILE(64, 128, 3) SWEEP_TILE(128, 64, 3) SWEEP_TILE(128, 128, 2)
#undef SWEEP_TILE
                printf("\n");
            }
            if (mfast)
            {   // the default tile pick, the grid order is the only change: same bits expected
                bf16* dYm; CK(cudaMalloc(&dYm, (size_t)M * N * G * 2));
                float us2[2] = {0, 0};
                for (int mf = 0; mf < 2; mf++)
                {
                    g_use_tc = true; g_tc_mfast = mf != 0;
                    bf16* Y = mf ? dYm : dYt;
                    launch_gemm(dA, lda, dsA, lsa, W, Y, ldy, M, G, K, KB, N, 0);
                    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                    cudaEventRecord(e0);
                    for (int r = 0; r < reps; r++) launch_gemm(dA, lda, dsA, lsa, W, Y, ldy, M, G, K, KB, N, 0);
                    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
                    float ms; cudaEventElapsedTime(&ms, e0, e1);
                    us2[mf] = ms * 1000.0f / reps;
                }
                g_use_tc = false; g_tc_mfast = false;
                std::vector<uint16_t> ym((size_t)M * N * G), yd((size_t)M * N * G);
                CK(cudaMemcpy(ym.data(), dYm, ym.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(yd.data(), dYt, yd.size() * 2, cudaMemcpyDeviceToHost));
                long md = 0; for (size_t i = 0; i < ym.size(); i++) md += ym[i] != yd[i];
                if (md) bad++;
                printf("mfast %-10s M %4d | N-fastest %.1f us %.1f TF/s | M-fastest %.1f us %.1f TF/s | x%.2f | %ld of %zu outputs differ %s\n", S.nm, M, us2[0], fl / (us2[0] * 1e6), us2[1], fl / (us2[1] * 1e6), us2[0] / us2[1], md, ym.size(), md ? "BAD" : "");
                cudaFree(dYm);
            }
            cudaFree(dA); cudaFree(dsA); cudaFree(dW); cudaFree(dsW); cudaFree(dYs); cudaFree(dYt);
        }
    printf("TC_GEMM_TEST %s (%d bad)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
