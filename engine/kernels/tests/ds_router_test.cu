// DS router (ds_router.cuh): CPU double reference, constructed edge cases, and GPU kernels vs the reference.
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -o ds_router_test ds_router_test.cu
//   CUDA_VISIBLE_DEVICES= ./ds_router_test host     reference self-checks only (no GPU)
//   ./ds_router_test gpu                            k_gate_logits and k_plan against the reference (a GPU that runs sm_121a)
//   ./ds_router_test bench                          k_gate_logits (cold L2) and k_plan timing
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include "../ds_router.cuh"
using namespace dsr;
// Geometry under test: the main model by default; -DDSR_NE=128 -DDSR_TK=3 = the DSpark draft layers 40..42.
#ifndef DSR_NE
#define DSR_NE 384
#endif
#ifndef DSR_TK
#define DSR_TK 6
#endif
constexpr int NE = DSR_NE, TK = DSR_TK, SM_ = MMAX * TK;
using TPlan = PlanT<TK>;
static constexpr int E_(int e) { return e % NE; }       // expert ids below were written for 384 experts; E_ maps them into range (identity at 384)


#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static double sqrt_softplus_d(double x) { return std::sqrt(x > 20.0 ? x : std::log1p(std::exp(x))); }

struct RefTok { int e[8]; double w[8]; double sc[8]; double gap; };   // selection order, best key first
static RefTok ref_token(const double* logit, const float* bias)
{
    std::vector<double> sc(NE), key(NE);
    for (int e = 0; e < NE; ++e) { sc[e] = sqrt_softplus_d(logit[e]); key[e] = sc[e] + static_cast<double>(bias[e]); }
    std::vector<int> id(NE);
    std::iota(id.begin(), id.end(), 0);
    std::stable_sort(id.begin(), id.end(), [&](int a, int b) { return key[a] > key[b]; });   // stable: a tie keeps the lowest id first
    RefTok r;
    double sum = 0;
    for (int k = 0; k < TK; ++k) { r.e[k] = id[k]; r.sc[k] = sc[id[k]]; sum += sc[id[k]]; }
    for (int k = 0; k < TK; ++k) r.w[k] = r.sc[k] / (sum + 1e-20) * 1.5;
    r.gap = key[id[TK - 1]] - key[id[TK]];
    return r;
}

struct Stats { int trials = 0, near_tie = 0, fails = 0; double max_w = 0; };

// TPlan invariants that hold for any input, then the comparison with the reference (set + weights; the order too when `ordered`).
static bool check_plan(const TPlan& P, int M, const std::vector<RefTok>& ref, bool ordered, Stats& st, const char* tag)
{
    ++st.trials;
    auto fail = [&](const char* what, int a, int b) { printf("  FAIL [%s] %s (%d, %d)\n", tag, what, a, b); ++st.fails; return false; };
    bool near = false;
    for (int m = 0; m < M; ++m) near |= ref[m].gap < 2e-5;
    if (P.n_slots < 1 || P.n_slots > M * TK) return fail("n_slots range", P.n_slots, M);
    for (int s = 0; s < SM_; ++s)
    {
        if (s < P.n_slots)
        {
            if (P.slot_expert[s] < 0 || P.slot_expert[s] >= NE) return fail("slot_expert range", s, P.slot_expert[s]);
            if (s && P.slot_expert[s] <= P.slot_expert[s - 1]) return fail("slot_expert not ascending", s, P.slot_expert[s]);
            if (P.slot_mask[s] == 0 || (P.slot_mask[s] >> M) != 0) return fail("slot_mask", s, P.slot_mask[s]);
        }
        else if (P.slot_expert[s] != -1 || P.slot_mask[s] != 0) return fail("unused slot not cleared", s, P.slot_expert[s]);
        for (int c = 0; c < 8; ++c)
        {
            const bool used = s < P.n_slots && (P.slot_mask[s] >> c & 1);
            if (used != (P.slot_tok[s][c] == c)) return fail("slot_tok vs mask", s, c);
            if (!used && (P.slot_tok[s][c] != -1 || P.slot_w[s][c] != 0.f)) return fail("unused column not cleared", s, c);
        }
    }
    for (int m = 0; m < MMAX; ++m)
        for (int k = 0; k < 8; ++k)
        {
            if (m >= M || k >= TK)
            {
                if (P.tok_slot[m][k] != -1 || P.tok_expert[m][k] != -1 || P.tok_w[m][k] != 0.f) return fail("dead route not cleared", m, k);
                continue;
            }
            const int s = P.tok_slot[m][k];
            if (s < 0 || s >= P.n_slots) return fail("tok_slot range", m, k);
            if (P.slot_expert[s] != P.tok_expert[m][k] || P.tok_col[m][k] != m) return fail("tok_slot vs slot_expert", m, k);
            if (!(P.slot_mask[s] >> m & 1) || P.slot_w[s][m] != P.tok_w[m][k]) return fail("slot_w vs tok_w", m, k);
        }
    for (int m = 0; m < M; ++m)
    {
        double sum = 0;
        for (int k = 0; k < TK; ++k)
        {
            sum += P.tok_w[m][k];
            for (int j = 0; j < k; ++j) if (P.tok_expert[m][j] == P.tok_expert[m][k]) return fail("duplicate route", m, k);
        }
        if (std::fabs(sum - 1.5) > 2e-5 && sum > 0) { if (ref[m].sc[0] > 1e-6) return fail("weights do not sum to 1.5", m, 0); }
    }
    if (near) { ++st.near_tie; return true; }       // two keys within rounding: the set may legitimately differ, invariants were checked
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < TK; ++k)
        {
            int kk = -1;
            for (int j = 0; j < TK; ++j) if (P.tok_expert[m][j] == ref[m].e[k]) kk = j;
            if (kk < 0) return fail("route set differs from reference", m, ref[m].e[k]);
            if (ordered && kk != k) return fail("route order differs from reference", m, k);
            if (ref[m].sc[k] > 1e-6)
            {
                const double rel = std::fabs(P.tok_w[m][kk] - ref[m].w[k]) / ref[m].w[k];
                st.max_w = std::max(st.max_w, rel);
                if (rel > 2e-5) return fail("weight differs from reference", m, ref[m].e[k]);
            }
        }
    return true;
}

// ----------------------------------------------------------------------------------------------- host-only checks of the reference
static int host_checks()
{
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { printf("  %-62s %s\n", what, ok ? "ok" : "FAIL"); bad += !ok; };
    static_assert(sizeof(TPlan) == 16 + 4 * SM_ * 2 + 4 * SM_ * 8 * 2 + 4 * MMAX * 8 * 4, "TPlan layout");
    std::vector<double> lg(NE, 0.0);
    std::vector<float> bias(NE, 0.f);
    RefTok r = ref_token(lg.data(), bias.data());
    bool asc = true; for (int k = 0; k < TK; ++k) asc &= r.e[k] == k;
    expect(asc, "all keys tied -> experts 0..TK-1 (lowest ids)");
    expect(std::fabs(r.w[0] - 1.5 / TK) < 1e-12, "TK equal scores -> weight 1.5 / TK each");
    for (int e = 0; e < NE; ++e) lg[e] = -1e4;
    for (int e : {E_(300), 17, E_(200)}) lg[e] = 25.0;                                  // above 20: the score is sqrt(x), no softplus
    bias[17] = 0.5f;
    r = ref_token(lg.data(), bias.data());
    expect(r.e[0] == 17 && r.e[1] == std::min(E_(200), E_(300)) && r.e[2] == std::max(E_(200), E_(300)) && (TK == 3 || r.e[3] == 0),
           "logits > 20 and bias: 17 (biased) first, then the other two (lowest id first), then ties 0, 1, 2");
    expect(std::fabs(r.sc[1] - 5.0) < 1e-12, "x = 25 -> score exactly sqrt(25) = 5");
    expect(std::fabs(sqrt_softplus_d(20.0) - sqrt_softplus_d(20.000001)) < 1e-6 && sqrt_softplus_d(-1e4) == 0.0,
           "score is continuous at x = 20, and x = -1e4 gives score 0");
    for (int e = 0; e < NE; ++e) { lg[e] = 0.0; bias[e] = 0.f; }
    bias[E_(123)] = 10.f;                                                            // a big bias forces the pick, the weight stays unbiased
    r = ref_token(lg.data(), bias.data());
    expect(r.e[0] == E_(123) && std::fabs(r.sc[0] - sqrt_softplus_d(0.0)) < 1e-15, "bias selects expert 123 (mod NE), its weight uses the unbiased score");
    printf("host: %s\n", bad ? "FAIL" : "OK");
    return bad;
}

// ----------------------------------------------------------------------------------------------- GPU
struct Dev
{
    float *logits, *bias; TPlan* plan; __nv_bfloat16 *gate, *x;
    Dev()
    {
        CK(cudaMalloc(&logits, MMAX * NE * 4)); CK(cudaMalloc(&bias, NE * 4)); CK(cudaMalloc(&plan, sizeof(TPlan)));
        CK(cudaMalloc(&gate, static_cast<size_t>(NE) * HIDDEN * 2)); CK(cudaMalloc(&x, static_cast<size_t>(MMAX) * HIDDEN * 2));
    }
};
static TPlan run_plan(Dev& d, const std::vector<float>& logits, const std::vector<float>& bias, int M)
{
    CK(cudaMemcpy(d.logits, logits.data(), static_cast<size_t>(M) * NE * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.bias, bias.data(), NE * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d.plan, 0x5a, sizeof(TPlan)));         // junk: every field must be written
    k_plan<NE, TK><<<1, PLAN_THREADS>>>(d.logits, d.bias, M, d.plan);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    TPlan P; CK(cudaMemcpy(&P, d.plan, sizeof(TPlan), cudaMemcpyDeviceToHost));
    return P;
}
static std::vector<RefTok> ref_all(const std::vector<float>& logits, const std::vector<float>& bias, int M)
{
    std::vector<RefTok> r;
    for (int m = 0; m < M; ++m)
    {
        std::vector<double> l(NE);
        for (int e = 0; e < NE; ++e) l[e] = logits[m * NE + e];
        r.push_back(ref_token(l.data(), bias.data()));
    }
    return r;
}

static int gpu_checks()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { printf("gpu: no CUDA device\n"); return 2; }
    Dev d;
    int bad = 0;
    std::mt19937 g(11);
    std::normal_distribution<float> nd(0.f, 1.f);

    // 1. k_plan on constructed edge cases (exact order expected: every input is a tie or far apart)
    {
        Stats st;
        for (int M : {1, 3, 8})
        {
            std::vector<float> lg(static_cast<size_t>(M) * NE, 0.f), bias(NE, 0.f);
            TPlan P = run_plan(d, lg, bias, M);
            check_plan(P, M, ref_all(lg, bias, M), true, st, "all tied");
            bool ok = P.n_slots == TK;
            for (int k = 0; k < TK; ++k) ok &= P.slot_expert[k] == k && P.slot_mask[k] == (1 << M) - 1;
            if (!ok) { printf("  FAIL all tied M=%d: slots are not experts 0..TK-1 with every token\n", M); ++st.fails; }
        }
        {   // three experts tie for the last place (ids 40, 41, 42 at the same key): the lowest id wins; other five fixed by a big bias
            std::vector<float> lg(NE, -50.f), bias(NE, 0.f);
            { const int big[5] = {7, 99, E_(250), E_(301), E_(383)}; for (int j = 0; j < TK - 1; ++j) bias[big[j]] = 5.f + 0.1f * big[j]; }   // TK - 1 sure picks
            for (int e : {40, 41, 42}) bias[e] = 2.f;
            TPlan P = run_plan(d, lg, bias, 1);
            check_plan(P, 1, ref_all(lg, bias, 1), true, st, "tie at the cut");
            bool has40 = false, has41 = false;
            for (int k = 0; k < TK; ++k) { has40 |= P.tok_expert[0][k] == 40; has41 |= P.tok_expert[0][k] == 41; }
            if (!has40 || has41) { printf("  FAIL tie at the cut: expected 40 in, 41 out\n"); ++st.fails; }
        }
        {   // logits above 20 (score = sqrt(x)) next to softplus ones, and large negatives
            std::vector<float> lg(NE, -1e4f), bias(NE, 0.f);
            lg[E_(300)] = 25.f; lg[17] = 30.f; lg[E_(200)] = 21.f; lg[3] = 20.f; lg[5] = 19.99f; lg[E_(380)] = 3.f;
            TPlan P = run_plan(d, lg, bias, 1);
            check_plan(P, 1, ref_all(lg, bias, 1), true, st, "logits > 20, big negatives");
        }
        {   // a biased expert is picked, its weight stays the unbiased score
            std::vector<float> lg(NE), bias(NE, 0.f);
            for (auto& v : lg) v = nd(g);
            bias[E_(123)] = 50.f; lg[E_(123)] = -3.f;
            TPlan P = run_plan(d, lg, bias, 1);
            check_plan(P, 1, ref_all(lg, bias, 1), true, st, "bias effect");
        }
        printf("constructed cases: %d checks, %d FAIL, max weight rel err %.2e\n", st.trials, st.fails, st.max_w);
        bad += st.fails;
    }

    // 2. k_plan on random logits, M = 1..8
    {
        Stats st;
        for (int M = 1; M <= 8; ++M)
            for (int trial = 0; trial < 40; ++trial)
            {
                std::vector<float> lg(static_cast<size_t>(M) * NE), bias(NE);
                for (auto& v : lg) v = nd(g) * 2.5f;
                for (auto& v : bias) v = nd(g) * 0.15f;
                if (trial % 5 == 4) for (int i = 0; i < 12; ++i) lg[(g() % M) * NE + g() % NE] = 20.f + 15.f * (g() % 100) / 100.f;
                TPlan P = run_plan(d, lg, bias, M);
                check_plan(P, M, ref_all(lg, bias, M), false, st, "random");
            }
        printf("random plans, M = 1..8: %d checks, %d near-tie (invariants only), %d FAIL, max weight rel err %.2e\n", st.trials, st.near_tie, st.fails, st.max_w);
        bad += st.fails;
    }

    // 3. k_gate_logits vs a double dot product, determinism, then gate -> plan end to end
    {
        std::vector<__nv_bfloat16> gate(static_cast<size_t>(NE) * HIDDEN), x(static_cast<size_t>(MMAX) * HIDDEN);
        for (auto& v : gate) v = __float2bfloat16_rn(nd(g) * 0.03f);
        for (auto& v : x) v = __float2bfloat16_rn(nd(g) * ((g() % 97 == 0) ? 40.f : 1.f));
        CK(cudaMemcpy(d.gate, gate.data(), gate.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        std::vector<float> bias(NE);
        for (auto& v : bias) v = nd(g) * 0.05f;
        for (int M : {1, 5, 8})
        {
            k_gate_logits<NE><<<NE / 8, 256>>>(d.gate, d.x, M, d.logits);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            std::vector<float> lg(static_cast<size_t>(M) * NE), lg2(lg.size());
            CK(cudaMemcpy(lg.data(), d.logits, lg.size() * 4, cudaMemcpyDeviceToHost));
            k_gate_logits<NE><<<NE / 8, 256>>>(d.gate, d.x, M, d.logits);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(lg2.data(), d.logits, lg2.size() * 4, cudaMemcpyDeviceToHost));
            const bool det = !memcmp(lg.data(), lg2.data(), lg.size() * 4);
            double worst = 0;
            std::vector<float> lgd(lg.size());
            for (int m = 0; m < M; ++m)
                for (int e = 0; e < NE; ++e)
                {
                    double ref = 0, mag = 0;
                    for (int k = 0; k < HIDDEN; ++k)
                    {
                        const double p = static_cast<double>(__bfloat162float(gate[static_cast<size_t>(e) * HIDDEN + k])) * __bfloat162float(x[static_cast<size_t>(m) * HIDDEN + k]);
                        ref += p; mag += std::fabs(p);
                    }
                    worst = std::max(worst, std::fabs(lg[m * NE + e] - ref) / (1e-5 * mag + 1e-7));
                    lgd[m * NE + e] = static_cast<float>(ref);
                }
            printf("k_gate_logits M=%d: determinism %s, max err / (1e-5 sum|w x|) = %.3f -> %s\n", M, det ? "OK" : "FAIL", worst, (det && worst <= 1.0) ? "OK" : "FAIL");
            bad += !(det && worst <= 1.0);
            Stats st;
            TPlan P = run_plan(d, lg, bias, M);          // the plan from the GPU's own logits, against the reference on the exact double logits
            check_plan(P, M, ref_all(lgd, bias, M), false, st, "end to end");
            printf("  gate -> plan M=%d: %d FAIL%s\n", M, st.fails, st.near_tie ? " (near-tie, invariants only)" : "");
            bad += st.fails;
        }
    }

    printf("gpu: %s\n", bad ? "FAIL" : "OK");
    return bad;
}

__global__ void k_empty() {}
// k_gate_logits on a rotating pool of 16 gate matrices (63 MB > the 24 MB L2, so every call reads DRAM), and k_plan alone.
static int bench()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { printf("bench: no CUDA device\n"); return 2; }
    constexpr int POOL = 16;
    const size_t gb = static_cast<size_t>(NE) * HIDDEN * 2;
    __nv_bfloat16 *gate, *x; float *logits, *bias; TPlan* plan;
    CK(cudaMalloc(&gate, gb * POOL)); CK(cudaMalloc(&x, static_cast<size_t>(MMAX) * HIDDEN * 2));
    CK(cudaMalloc(&logits, MMAX * NE * 4)); CK(cudaMalloc(&bias, NE * 4)); CK(cudaMalloc(&plan, sizeof(TPlan)));
    CK(cudaMemset(gate, 0x3c, gb * POOL)); CK(cudaMemset(x, 0x3c, static_cast<size_t>(MMAX) * HIDDEN * 2));
    CK(cudaMemset(logits, 0, MMAX * NE * 4)); CK(cudaMemset(bias, 0, NE * 4));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time_us = [&](auto&& f) {
        for (int i = 0; i < 2 * POOL; ++i) f(i);
        CK(cudaDeviceSynchronize());
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep)
        {
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 20 * POOL; ++i) f(i);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms * 1000.0 / (20 * POOL));
        }
        return best;
    };
    printf("gate bf16 [%d][%d] = %.2f MB per layer; ceiling at 255.8 GB/s = %.1f us, at 273 GB/s = %.1f us\n", NE, HIDDEN, gb / 1e6, gb / 255.8e3, gb / 273e3);
    for (int M : {1, 8})
    {
        const double us = time_us([&](int i) { k_gate_logits<NE><<<NE / 8, 256>>>(gate + (i % POOL) * (gb / 2), x, M, logits); });
        printf("  k_gate_logits M=%d   %6.1f us  %6.1f GB/s  %5.1f%% of 255.8\n", M, us, gb / us / 1e3, 100 * gb / us / 1e3 / 255.8);
    }
    printf("  empty kernel         %6.1f us (launch floor, back to back)\n", time_us([&](int) { k_empty<<<1, PLAN_THREADS>>>(); }));
    for (int M : {0, 1, 8})
    {
        const double us = time_us([&](int) { k_plan<NE, TK><<<1, PLAN_THREADS>>>(logits, bias, M, plan); });
        printf("  k_plan M=%d          %6.1f us (one CTA, back to back)\n", M, us);
    }
#ifdef DSR_PROF
    for (int M : {1})
    {
        k_plan<NE, TK><<<1, PLAN_THREADS>>>(logits, bias, M, plan); CK(cudaDeviceSynchronize());
        k_plan<NE, TK><<<1, PLAN_THREADS>>>(logits, bias, M, plan); CK(cudaDeviceSynchronize());
        long long c[8]; CK(cudaMemcpyFromSymbol(c, g_dsr_prof, sizeof(c)));
        printf("  k_plan M=%d warp-0 cycles: init+scoring %lld, top-6 %lld, rest (weights, slots) %lld (total %lld)\n", M, c[1] - c[0], c[2] - c[1], c[3] - c[2], c[3] - c[0]);
    }
#endif
    return 0;
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "host";
    if (!strcmp(mode, "host")) return host_checks() ? 1 : 0;
    if (!strcmp(mode, "gpu")) return gpu_checks() ? 1 : 0;
    if (!strcmp(mode, "bench")) return bench();
    printf("usage: %s host|gpu|bench\n", argv[0]);
    return 2;
}
