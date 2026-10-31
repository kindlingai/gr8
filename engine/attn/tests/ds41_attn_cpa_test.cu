// Bitwise A/B test of the cp.async attention core (k_attn_cpa, GR8_DS_ATTN_CPA=1) against k_attn: same inputs, outputs compared with memcmp.
// No fixture, no torch: random bf16 q / window ring / compressed rows / top-k lists, every config run on both kernels (and once through
// launch_attn with g_attn_cpa toggled, which checks the dispatch).
//
//   build: nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I engine -o ds41_attn_cpa_test engine/attn/tests/ds41_attn_cpa_test.cu engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm
//   run:   ds41_attn_cpa_test [--time ITERS]      (one Spark GPU; seconds)
//
// Configs: decode (ring window, WL = 128, ring_rows 128 / 135, host start and the device start word) and prefill (WL = min(n, 128)) for
// n = 1..8 (+ 130 prefill), TK in {0, 1, 63, 64, 65, 200, 511, 512}, top-k lists full / with a -1 tail / with scattered -1 / with a whole
// dead block in the middle / all -1, heads 16, 32 and 96 (the 6-CTA shape of a TP4 rank). Exit 0 = every byte equal.
#include "../ds41_attn.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ds41a
{

#define CK(x)                                                                                                       \
    do                                                                                                              \
    {                                                                                                               \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess)                                                                                      \
        {                                                                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                \
            exit(2);                                                                                                \
        }                                                                                                           \
    } while (0)

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static float urand() { return (float)(rnd() >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f; }
static std::vector<bf16> rand_bf16(size_t n, float amp)
{
    std::vector<bf16> v(n);
    for (auto& x : v) x = __float2bfloat16_rn(urand() * amp);
    return v;
}
template <class T> static T* upload(const std::vector<T>& h)
{
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

enum { TKL_FULL, TKL_TAIL, TKL_SCATTER, TKL_HOLE, TKL_NONE, TKL_N };
static const char* tkl_name[TKL_N] = {"full", "tail-1", "scatter-1", "hole-block", "all-1"};

int main_test(int argc, char** argv)
{
    int titers = 0;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--time") && i + 1 < argc) titers = atoi(argv[++i]);
    CK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)AT_SMEM));
    CK(cudaFuncSetAttribute(k_attn_cpa, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)AT_SMEM_CPA));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));

    const int NCOMP = 1024, RMAX = WIN + 8, NMAX = 130, KLD = 512;
    const std::vector<bf16> hck = rand_bf16((size_t)NCOMP * HD, 1.5f);
    const std::vector<bf16> hring = rand_bf16((size_t)(NMAX > RMAX ? NMAX : RMAX) * HD, 1.5f);
    bf16* dck = upload(hck);
    bf16* dring = upload(hring);   // decode: the ring [ring_rows][512]; prefill: this chunk's rows [n][512]
    const int heads_list[3] = {16, 32, 96};
    const int ns[9] = {1, 2, 3, 4, 5, 6, 7, 8, 130};
    const int tks[8] = {0, 1, 63, 64, 65, 200, 511, 512};
    int configs = 0, bad = 0, nlaunch_mismatch = 0;
    size_t bytes_bad = 0;
    for (int hi = 0; hi < 3; hi++)
    {
        const int heads = heads_list[hi];
        const std::vector<bf16> hq = rand_bf16((size_t)NMAX * heads * HD, 2.0f);
        std::vector<float> hsink(heads);
        for (auto& s : hsink) s = urand() * 2.0f;
        bf16* dq = upload(hq);
        float* dsink = upload(hsink);
        bf16* dout_a = nullptr;
        bf16* dout_b = nullptr;
        const size_t obytes = (size_t)NMAX * heads * HD * sizeof(bf16);
        CK(cudaMalloc(&dout_a, obytes));
        CK(cudaMalloc(&dout_b, obytes));
        std::vector<bf16> ha(NMAX * (size_t)heads * HD), hb(NMAX * (size_t)heads * HD);
        for (int ni = 0; ni < 9; ni++)
        {
            const int n = ns[ni];
            if (n == 130 && hi != 0) continue;   // the big prefill chunk once, at heads 16
            for (int mode = 0; mode < 2; mode++)   // 0 = decode (ring), 1 = prefill
            {
                if (n == 130 && mode == 0) continue;   // decode rows are <= 8
                const int ring_variants = mode == 0 ? 2 : 1;
                for (int rv = 0; rv < ring_variants; rv++)
                for (int ti = 0; ti < 8; ti++)
                {
                    const int TK = tks[ti];
                    for (int kind = 0; kind < TKL_N; kind++)
                    {
                        if (TK == 0 && kind > 0) continue;
                        std::vector<int> tk((size_t)NMAX * KLD, -1);
                        for (int r = 0; r < n; r++)
                            for (int j = 0; j < TK; j++)
                            {
                                int v = (int)(rnd() % NCOMP);
                                if (kind == TKL_TAIL && j >= TK / 2) v = -1;
                                if (kind == TKL_SCATTER && rnd() % 5 == 0) v = -1;
                                if (kind == TKL_HOLE && j >= 64 - 0 && j < 128) v = -1;   // slots WL+64..WL+127 dead: a middle block when WL = 128
                                if (kind == TKL_NONE) v = -1;
                                tk[(size_t)r * KLD + j] = v;
                            }
                        int* dtk = upload(tk);
                        const int starts[4] = {1, 5, 127, 1000};
                        const int nstart = mode == 0 ? 4 : 1;
                        for (int si = 0; si < nstart; si++)
                        {
                            for (int dev = 0; dev < (mode == 0 ? 2 : 1); dev++)
                            {
                                AttnP a{};
                                a.q = dq;
                                a.kvw = dring;
                                a.ck = TK ? dck : nullptr;
                                a.topk = TK ? dtk : nullptr;
                                a.sink = dsink;
                                a.ldtk = KLD;
                                a.n = n;
                                a.heads = heads;
                                a.decode = mode == 0 ? 1 : 0;
                                a.WL = mode == 0 ? WIN : (n < WIN ? n : WIN);
                                a.ring_rows = mode == 0 ? (rv ? WIN + 7 : WIN) : 0;
                                a.TK = TK;
                                a.scale = 0.044194174f;   // 512^-0.5
                                int* dstart = nullptr;
                                if (mode == 0)
                                {
                                    if (dev) { CK(cudaMalloc(&dstart, sizeof(int))); CK(cudaMemcpy(dstart, &starts[si], sizeof(int), cudaMemcpyHostToDevice)); a.dstart = dstart; a.start = 0; }
                                    else a.start = starts[si];
                                }
                                CK(cudaMemset(dout_a, 0xFF, obytes));
                                CK(cudaMemset(dout_b, 0xFF, obytes));
                                const dim3 grid(n, heads / 16);
                                a.out = dout_a;
                                k_attn<<<grid, AT_THREADS, AT_SMEM, st>>>(a);
                                a.out = dout_b;
                                k_attn_cpa<<<grid, AT_THREADS, AT_SMEM_CPA, st>>>(a);
                                CK(cudaStreamSynchronize(st));
                                CK(cudaGetLastError());
                                const size_t nb = (size_t)n * heads * HD * sizeof(bf16);
                                CK(cudaMemcpy(ha.data(), dout_a, nb, cudaMemcpyDeviceToHost));
                                CK(cudaMemcpy(hb.data(), dout_b, nb, cudaMemcpyDeviceToHost));
                                configs++;
                                if (memcmp(ha.data(), hb.data(), nb) != 0)
                                {
                                    size_t d = 0;
                                    for (size_t i = 0; i < nb; i++) d += ((const uint8_t*)ha.data())[i] != ((const uint8_t*)hb.data())[i];
                                    bad++;
                                    bytes_bad += d;
                                    if (bad <= 20)
                                        fprintf(stderr, "MISMATCH heads=%d n=%d %s ring=%d TK=%d list=%s start=%d dev=%d: %zu bytes differ\n", heads, n,
                                                mode ? "prefill" : "decode", a.ring_rows, TK, tkl_name[kind], mode ? 0 : starts[si], dev, d);
                                }
                                // the dispatch: launch_attn with g_attn_cpa = 1 must equal the cpa kernel, = 0 k_attn
                                if (si == 0 && dev == 0 && kind == 0)
                                {
                                    for (int sel = 0; sel < 2; sel++)
                                    {
                                        g_attn_cpa = sel;
                                        a.out = dout_b;
                                        CK(cudaMemset(dout_b, 0xFF, obytes));
                                        launch_attn(a, st);
                                        CK(cudaStreamSynchronize(st));
                                        CK(cudaMemcpy(hb.data(), dout_b, nb, cudaMemcpyDeviceToHost));
                                        if (memcmp(ha.data(), hb.data(), nb) != 0) { nlaunch_mismatch++; fprintf(stderr, "launch_attn(g_attn_cpa=%d) differs, n=%d TK=%d\n", sel, n, TK); }
                                    }
                                    g_attn_cpa = 0;
                                }
                                if (dstart) CK(cudaFree(dstart));
                            }
                        }
                        CK(cudaFree(dtk));
                    }
                }
            }
        }
        // timing: n = 1 and 6 rows, 96 heads (the engine's grids), per kernel, us per launch
        if (titers > 0 && hi == 2)
        {
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            for (int nn : {1, 6})
                for (int TK : {0, 64, 200, 512})
                {
                    std::vector<int> tk((size_t)NMAX * KLD, -1);
                    for (int r = 0; r < nn; r++)
                        for (int j = 0; j < TK; j++) tk[(size_t)r * KLD + j] = (int)(rnd() % NCOMP);
                    int* dtk = upload(tk);
                    int hs = 1000;
                    int* dstart = nullptr;
                    CK(cudaMalloc(&dstart, sizeof(int)));
                    CK(cudaMemcpy(dstart, &hs, sizeof(int), cudaMemcpyHostToDevice));
                    AttnP a{};
                    a.q = dq; a.kvw = dring; a.ck = TK ? dck : nullptr; a.topk = TK ? dtk : nullptr; a.sink = dsink; a.out = dout_a;
                    a.ldtk = KLD; a.n = nn; a.heads = 96; a.decode = 1; a.WL = WIN; a.ring_rows = WIN + 7; a.TK = TK; a.scale = 0.044194174f;
                    a.dstart = dstart;
                    const dim3 grid(nn, 96 / 16);
                    float us[2] = {0, 0};
                    for (int k = 0; k < 2; k++)
                    {
                        for (int w = 0; w < 20; w++)
                        {
                            if (k) k_attn_cpa<<<grid, AT_THREADS, AT_SMEM_CPA, st>>>(a);
                            else k_attn<<<grid, AT_THREADS, AT_SMEM, st>>>(a);
                        }
                        CK(cudaStreamSynchronize(st));
                        CK(cudaEventRecord(e0, st));
                        for (int w = 0; w < titers; w++)
                        {
                            if (k) k_attn_cpa<<<grid, AT_THREADS, AT_SMEM_CPA, st>>>(a);
                            else k_attn<<<grid, AT_THREADS, AT_SMEM, st>>>(a);
                        }
                        CK(cudaEventRecord(e1, st));
                        CK(cudaEventSynchronize(e1));
                        float ms = 0;
                        CK(cudaEventElapsedTime(&ms, e0, e1));
                        us[k] = ms * 1000.0f / titers;
                    }
                    printf("TIME n=%d heads=96 WL=128 TK=%3d: k_attn %.2f us   k_attn_cpa %.2f us   (%.2fx)\n", nn, TK, us[0], us[1], us[0] / us[1]);
                    CK(cudaFree(dtk));
                    CK(cudaFree(dstart));
                }
        }
        CK(cudaFree(dq));
        CK(cudaFree(dsink));
        CK(cudaFree(dout_a));
        CK(cudaFree(dout_b));
    }
    printf("ATTN_CPA_TEST: configs=%d mismatched=%d bytes_differ=%zu launch_attn_mismatch=%d  => %s\n", configs, bad, bytes_bad, nlaunch_mismatch,
           (bad || nlaunch_mismatch) ? "FAIL" : "PASS (every output byte equal)");
    return (bad || nlaunch_mismatch) ? 1 : 0;
}

}   // namespace ds41a

int main(int argc, char** argv) { return ds41a::main_test(argc, argv); }
