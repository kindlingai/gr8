// KDA prefill scan bench + check on synthetic inputs (no weights): per-kernel time of launch_core_prefill (prep, intra,
// state, out) at M rows of one slot, nh heads, and the chunked result (OG fp32 gated o, final S, conv history) vs the
// decode core kernel (k_kda_core, row by row) from the same inputs and start state.
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -o kda_pf_bench kda_pf_bench.cu
//   kda_pf_bench [M=9216] [nh=32] [reps=5]
// After the OFF variants: the G9 ON arm (kda::g9::launch, VS 1/2/4, one total time each; writes OH fp16, compared with the
// decode core OG).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "../kda_g9.cuh"

using namespace kda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

template <class T> static T* dalloc(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T))); CK(cudaMemset(d, 0, n * sizeof(T))); return d; }
template <class T> static T* up(const std::vector<T>& h) { T* d = dalloc<T>(h.size()); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static double rel(const std::vector<float>& a, const std::vector<float>& b, double* maxabs = nullptr)
{
    double d = 0, r = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const double e = static_cast<double>(a[i]) - b[i];
        d += e * e; r += static_cast<double>(b[i]) * b[i]; mx = std::max(mx, std::fabs(e));
    }
    if (maxabs) *maxabs = mx;
    return std::sqrt(d / (r > 0 ? r : 1));
}

int main(int argc, char** argv)
{
    const int M = argc > 1 ? atoi(argv[1]) : 9216, nh = argc > 2 ? atoi(argv[2]) : 32, reps = argc > 3 ? atoi(argv[3]) : 5;
    const int dl = nh * DK;
    PLayout pl; pl.dl = dl;
    const int ld = pl.ld();
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bf = [](float f) { return f2bf_rn(f); };

    // inputs: projections P (q/k/v unit normal; f_a, g_a, b scaled so G spans the gate range), weights, start state
    std::vector<float> hP(static_cast<size_t>(M) * ld);
    for (auto& x : hP) x = nd(rng);
    std::vector<uint16_t> hconv[3], hfb(static_cast<size_t>(dl) * DK), hgb(hfb.size()), hon(DK);
    for (int s = 0; s < 3; ++s) { hconv[s].resize(static_cast<size_t>(dl) * CONV); for (auto& x : hconv[s]) x = bf(0.5f * nd(rng)); }
    for (auto& x : hfb) x = bf(0.15f * nd(rng));
    for (auto& x : hgb) x = bf(0.1f * nd(rng));
    for (auto& x : hon) x = bf(1.f + 0.1f * nd(rng));
    std::vector<float> hAlog(nh), hdt(dl), hS(static_cast<size_t>(nh) * DK * DV), hcst(static_cast<size_t>(NHIST) * 3 * dl);
    for (auto& x : hAlog) x = 0.5f * nd(rng);
    for (auto& x : hdt) x = -2.f + nd(rng);
    for (auto& x : hS) x = 0.05f * nd(rng);
    for (auto& x : hcst) x = nd(rng);

    Args a;
    a.nh = nh; a.h0 = 0; a.M = M; a.nhmax = nh;
    const uint16_t* dconv[3];
    for (int s = 0; s < 3; ++s) dconv[s] = up(hconv[s]);
    for (int s = 0; s < 3; ++s) a.conv_w[s] = dconv[s];
    a.fb = up(hfb); a.gb = up(hgb); a.onorm = up(hon); a.A_log = up(hAlog); a.dt_bias = up(hdt);
    std::vector<int> hslot(M, 0);
    a.row_slot = up(hslot);
    float* S0 = up(hS);
    float* c0 = up(hcst);
    a.S = dalloc<float>(hS.size()); a.cst = dalloc<float>(hcst.size());
    a.P = up(hP);
    a.C = dalloc<float>(static_cast<size_t>(M) * 3 * dl); a.G = dalloc<float>(static_cast<size_t>(M) * dl);
    a.Z = dalloc<float>(static_cast<size_t>(M) * dl); a.O = dalloc<float>(static_cast<size_t>(M) * dl);
    a.OG = dalloc<float>(static_cast<size_t>(M) * dl);
    KdaPrefillScratch ps;
    if (const char* e = build_prefill_scratch(ps, M, 1)) { fprintf(stderr, "%s\n", e); return 3; }
    const PfArgs p = make_pf_args(a, ps);
    auto reset = [&] {
        CK(cudaMemcpy(a.S, S0, hS.size() * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(a.cst, c0, hcst.size() * 4, cudaMemcpyDeviceToDevice));
    };

    // reference: the decode core kernel row by row
    reset();
    k_kda_core<true><<<nh, CORE_T>>>(a);
    CK(cudaDeviceSynchronize());
    const auto rOG = down(a.OG, static_cast<size_t>(M) * dl), rS = down(a.S, hS.size()), rc = down(a.cst, hcst.size());

    // chunked prefill, timed per stage; variant 0 = row-loop prep + state v1 (old), 1 = row-parallel prep, 2 = + state v2, 3 = + intra v2, 4 = intra v3
    cudaEvent_t ev[5];
    for (auto& e : ev) CK(cudaEventCreate(&e));
    const int nc = (M + CH - 1) / CH;
    bool ok = true;
    for (int var = 0; var < 5; ++var)
    {
        double tk[4] = {0, 0, 0, 0};
        for (int r = 0; r <= reps; ++r)
        {
            reset();
            CK(cudaMemset(a.OG, 0, static_cast<size_t>(M) * dl * 4));
            CK(cudaEventRecord(ev[0]));
            if (var == 0) k_kda_prep<<<nh * 2, PREP_T>>>(a);
            else launch_prep_pf(a, 0);
            CK(cudaEventRecord(ev[1]));
            if (var < 3) k_kda_intra<<<dim3(nh, p.nchunk()), INTRA_T>>>(p);
            else if (var == 3) k_kda_intra2<<<dim3(nh, p.nchunk()), INTRA_T>>>(p);
            else k_kda_intra3<<<dim3(nh, p.nchunk()), INTRA_T>>>(p);
            CK(cudaEventRecord(ev[2]));
            if (var < 2) k_kda_state<<<dim3(nh, DV / VB), ST_T>>>(p);
            else k_kda_state2<<<dim3(nh, DV / VB), ST_T>>>(p);
            CK(cudaEventRecord(ev[3]));
            k_kda_out<<<dim3(nh, M), OUT_T>>>(a);
            CK(cudaEventRecord(ev[4]));
            CK(cudaDeviceSynchronize());
            if (r == 0) continue;   // warm-up
            for (int k = 0; k < 4; ++k) { float ms; CK(cudaEventElapsedTime(&ms, ev[k], ev[k + 1])); tk[k] += ms / reps; }
        }
        const auto cOG = down(a.OG, static_cast<size_t>(M) * dl), cS = down(a.S, hS.size()), cc = down(a.cst, hcst.size());
        double mx;
        printf("[var %d] M %d nh %d: prep %.3f ms  intra %.3f  state %.3f  out %.3f  total %.3f ms\n", var, M, nh, tk[0], tk[1], tk[2],
               tk[3], tk[0] + tk[1] + tk[2] + tk[3]);
        const double eo = rel(cOG, rOG, &mx);
        printf("  vs decode core: OG rel %.3e (max abs %.3e)", eo, mx);
        const double es = rel(cS, rS, &mx);
        printf("  S rel %.3e (max abs %.3e)", es, mx);
        const double ec = rel(cc, rc, &mx);
        printf("  conv hist rel %.3e\n", ec);
        double worst = 0; int wc = 0;
        for (int c = 0; c < nc; ++c)
        {
            double d = 0, r = 0;
            for (size_t i = static_cast<size_t>(c) * CH * dl; i < std::min<size_t>(static_cast<size_t>(c + 1) * CH, M) * dl; ++i)
            { const double e = cOG[i] - rOG[i]; d += e * e; r += static_cast<double>(rOG[i]) * rOG[i]; }
            const double x = std::sqrt(d / (r > 0 ? r : 1));
            if (x > worst) { worst = x; wc = c; }
        }
        printf("  worst chunk OG rel %.3e at chunk %d of %d\n", worst, wc, nc);
        ok = ok && eo < 2e-3 && es < 2e-3 && ec < 1e-6;
    }

    // G9 ON arm: planes live in a.C (M x 3dl fp32 = six bf16 planes of ps.rows = M rows); OH only here
    {
        const auto h2f = [](uint16_t h) {
            const int sg = h >> 15, e = (h >> 10) & 31, m = h & 1023;
            const float v = e == 0 ? std::ldexp(static_cast<float>(m), -24) : e == 31 ? INFINITY : std::ldexp(static_cast<float>(1024 + m), e - 25);
            return sg ? -v : v;
        };
        Args ao = a;
        ao.OH = dalloc<uint16_t>(static_cast<size_t>(M) * dl);
        for (int VS : {1, 2, 4})
        {
            double tot = 0;
            for (int r = 0; r <= reps; ++r)
            {
                reset();
                CK(cudaEventRecord(ev[0]));
                if (const char* e = kda::g9::launch(ao, ps, VS, 0)) { printf("[G9 ON VS %d] skipped: %s\n", VS, e); tot = -1; break; }
                CK(cudaEventRecord(ev[1]));
                CK(cudaDeviceSynchronize());
                CK(cudaGetLastError());
                if (r == 0) continue;
                float ms; CK(cudaEventElapsedTime(&ms, ev[0], ev[1])); tot += ms / reps;
            }
            if (tot < 0) continue;
            const auto hb = down(ao.OH, static_cast<size_t>(M) * dl);
            std::vector<float> cOH(hb.size());
            for (size_t i = 0; i < hb.size(); ++i) cOH[i] = h2f(hb[i]);
            const auto cS = down(a.S, hS.size()), cc = down(a.cst, hcst.size());
            double mx;
            const double eo = rel(cOH, rOG, &mx), es = rel(cS, rS, nullptr), ec = rel(cc, rc, nullptr);
            printf("[G9 ON VS %d] M %d nh %d: total %.3f ms  vs decode core: OH rel %.3e (max abs %.3e)  S rel %.3e  conv hist rel %.3e\n", VS, M, nh, tot, eo,
                   mx, es, ec);
            ok = ok && eo < 2e-2 && es < 2e-2 && ec < 1e-6;
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
