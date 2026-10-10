// rstest: the vocab-split head kernels (glue.cuh) of all NR ranks on ONE GPU with a loopback fabric, for rejection
// sampling (--spec-accept rejection). Synthetic logits, 8 rows: row m (m < 7) carries a draft, the step's
// input token of row m + 1 (as in a verify step); row 7 has none. Each seed is one step. Per mode:
//   - the output token frequencies of every row against p (softmax of logit / T, the top_p nucleus with --top-p-cands)
//     by chi-square, for the coupled kernels (today's) and the rejection kernels
//   - the share of seeds that kept the draft against p(draft), by a binomial z-test
//   - row 7 (no draft) and every row at T = 0: rejection kernels give the coupled kernels' bits
//   - every rank writes the same tokens
// rstest [--seeds N] [--temp T] [--top-p P]   exit 1 if a check fails (chi-square or binomial p < 0.001, or a bit difference)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
#include "../kernels/glue.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(2); } } while (0)
template <typename T> T* dalloc(size_t n) { T* p; CK(cudaMalloc((void**)&p, n * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); return p; }

constexpr int NR = g6nr::NR, M = 8, VN = 256, V = NR * VN;

struct Loop { const char* send[NR]; char* recv; const u64* stepp; int ncoll, n16; size_t stride; };
// the NIC: every rank's send slot of collective 0 -> the shared recv slot [rank], then the tail. grid NR, block 128.
__global__ void k_loop(Loop a)
{
    const u64 step = *a.stepp;
    const size_t slot = (step & 1) * a.ncoll;
    const uint4* s = reinterpret_cast<const uint4*>(a.send[blockIdx.x] + slot * a.stride);
    uint4* d = reinterpret_cast<uint4*>(a.recv + (slot * NR + blockIdx.x) * a.stride);
    for (int i = threadIdx.x; i < a.n16; i += 128) d[i] = s[i];
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); *reinterpret_cast<u64*>(d + a.n16) = g6::coll_seq(step, 0, a.ncoll); }
}
__global__ void k_set(u64* p, u64 v) { *p = v; }

// Q(a, x), the upper regularized incomplete gamma: the chi-square survival is Q(df / 2, chi2 / 2)
static double gammq(double a, double x)
{
    if (x <= 0) return 1;
    const double gl = std::lgamma(a);
    if (x < a + 1)
    {
        double ap = a, sum = 1 / a, del = sum;
        for (int n = 0; n < 1000 && std::fabs(del) > std::fabs(sum) * 1e-15; ++n) { ap += 1; del *= x / ap; sum += del; }
        return 1 - sum * std::exp(-x + a * std::log(x) - gl);
    }
    double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (int i = 1; i < 1000; ++i)
    {
        const double an = -i * (i - a);
        b += 2;
        d = an * d + b; if (std::fabs(d) < 1e-300) d = 1e-300;
        c = b + an / c; if (std::fabs(c) < 1e-300) c = 1e-300;
        d = 1 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1) < 1e-15) break;
    }
    return std::exp(-x + a * std::log(x) - gl) * h;
}

// p of one row as the kernels define it: softmax(logit / T) over the vocabulary, or over the nucleus (sorted by logit
// desc, id asc; keep while the mass before a token is < top_p of the whole) when top_p is on
static std::vector<double> target_p(const float* lg, float inv_temp, float top_p)
{
    std::vector<double> p(V, 0.0);
    const float mx = *std::max_element(lg, lg + V);
    double z = 0;
    for (int v = 0; v < V; ++v) z += std::exp((double)(lg[v] - mx) * inv_temp);
    std::vector<int> ix(V);
    std::iota(ix.begin(), ix.end(), 0);
    std::sort(ix.begin(), ix.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    double before = 0, kept = 0;
    for (int v : ix)
    {
        if (top_p > 0.f && top_p < 1.f && before >= top_p * z) break;
        const double w = std::exp((double)(lg[v] - mx) * inv_temp);
        p[v] = w; kept += w; before += w;
    }
    for (double& x : p) x /= kept;
    return p;
}

// chi-square of observed counts against N * p: bins with expectation >= 5, the rest pooled into one bin
static double chi2_p(const std::vector<long>& obs, const std::vector<double>& p, long N, int& df, long& outside)
{
    double chi = 0, eo = 0;
    long oo = 0;
    int bins = 0;
    outside = 0;
    for (int v = 0; v < V; ++v)
    {
        const double e = p[v] * N;
        if (p[v] == 0) { outside += obs[v]; continue; }
        if (e >= 5) { chi += (obs[v] - e) * (obs[v] - e) / e; ++bins; }
        else { eo += e; oo += obs[v]; }
    }
    if (eo > 0) { chi += (oo - eo) * (oo - eo) / eo; ++bins; }
    df = bins - 1;
    return df > 0 ? gammq(df / 2.0, chi / 2.0) : 1.0;
}

int main(int argc, char** argv)
{
    long N = 20000;
    float temp = 1.f, top_p = 0.9f;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--seeds") N = atol(argv[i + 1]);
        else if (a == "--temp") temp = (float)atof(argv[i + 1]);
        else if (a == "--top-p") top_p = (float)atof(argv[i + 1]);
        else { fprintf(stderr, "usage: rstest [--seeds N] [--temp T] [--top-p P]\n"); return 2; }
    }
    // logits: row m ranks the vocabulary by a permutation of its own (the top tokens spread over the ranks) and gives
    // rank k the logit -alpha_m * log(1 + k), so the rows run from flat to peaked
    const float alpha[M] = {1.2f, 1.6f, 2.0f, 1.4f, 6.0f, 2.5f, 1.8f, 1.5f};
    std::vector<float> lg((size_t)M * V);
    std::vector<int> order((size_t)M * V);
    srand(7);
    for (int m = 0; m < M; ++m)
    {
        int* o = &order[(size_t)m * V];
        std::iota(o, o + V, 0);
        for (int i = V - 1; i > 0; --i) std::swap(o[i], o[rand() % (i + 1)]);
        for (int k = 0; k < V; ++k) lg[(size_t)m * V + o[k]] = -alpha[m] * std::log1p((float)k);
    }
    // drafts: row m's draft = tok[m + 1]. Ranks in row m's order: top, second, fifth, far outside any nucleus, top of a
    // peaked row, tenth, third; row 7 has none
    const int dk[M - 1] = {0, 1, 4, V / 2, 0, 9, 2};
    int tok[M];
    tok[0] = 11;
    for (int m = 0; m < M - 1; ++m) tok[m + 1] = order[(size_t)m * V + dk[m]];

    const int ncs[2] = {0, 64};
    bool fail = false;
    for (int nc : ncs)
        for (float T : {temp, 0.f})
        {
            const float inv_temp = T > 0 ? 1.f / T : 0.f;
            const float tp = nc > 0 ? top_p : 0.f;
            const size_t pay = glue::head_bytes(M, nc), stride = (pay + 8 + 255) / 256 * 256;
            char* recv = dalloc<char>(2 * (size_t)NR * stride);
            u64* stepp = dalloc<u64>(1);
            unsigned* err = dalloc<unsigned>(1);
            unsigned long long* bell = dalloc<unsigned long long>(1);
            unsigned* miss = dalloc<unsigned>(1);
            int* tok_d = dalloc<int>(M);
            CK(cudaMemcpy(tok_d, tok, sizeof tok, cudaMemcpyHostToDevice));
            std::vector<glue::SampleRow> rows((size_t)N * M);
            for (long s = 0; s < N; ++s)
                for (int m = 0; m < M; ++m) rows[(size_t)s * M + m] = glue::SampleRow{inv_temp, (uint32_t)(s + 1), 0u, 1000 + m, 0, tp, 0};
            glue::SampleRow* rows_d = dalloc<glue::SampleRow>(rows.size());
            CK(cudaMemcpy(rows_d, rows.data(), rows.size() * sizeof(glue::SampleRow), cudaMemcpyHostToDevice));
            Loop loop{};
            loop.recv = recv; loop.stepp = stepp; loop.ncoll = 1; loop.n16 = (int)(pay / 16); loop.stride = stride;
            std::vector<glue::HeadArgs> A(NR);
            std::vector<int*> out(NR * 2);
            std::vector<float*> lpo(NR * 2);
            for (int r = 0; r < NR; ++r)
            {
                float* l = dalloc<float>((size_t)M * VN);
                std::vector<float> sl((size_t)M * VN);
                for (int m = 0; m < M; ++m) memcpy(&sl[(size_t)m * VN], &lg[(size_t)m * V + r * VN], VN * 4);
                CK(cudaMemcpy(l, sl.data(), sl.size() * 4, cudaMemcpyHostToDevice));
                g6::FabricDev f{};
                f.send = dalloc<char>(2 * stride);
                f.recv = recv; f.doorbell = bell; f.error = err;
                f.rank = r; f.nr = NR; f.ncoll = 1; f.stride = stride;
                loop.send[r] = f.send;
                glue::HeadArgs& a = A[r];
                a.f = f; a.stepp = stepp; a.c = 0; a.M = M; a.logits = l; a.ldl = VN; a.vlo = r * VN; a.vn = VN;
                a.ticket = dalloc<unsigned>(1); a.nc = nc; a.topp_miss = miss;
                for (int k = 0; k < 2; ++k) { out[r * 2 + k] = dalloc<int>((size_t)N * M); lpo[r * 2 + k] = dalloc<float>((size_t)N * M); }
            }
            u64 step = 0;
            for (int k = 0; k < 2; ++k)          // 0 = coupled kernels, 1 = rejection kernels
                for (long s = 0; s < N; ++s)
                {
                    k_set<<<1, 1>>>(stepp, ++step);
                    for (int r = 0; r < NR; ++r)
                    {
                        glue::HeadArgs a = A[r];
                        a.rows = rows_d + (size_t)s * M; a.tok_out = out[r * 2 + k] + (size_t)s * M; a.lp_out = lpo[r * 2 + k] + (size_t)s * M;
                        if (k) glue::k_head_local_rs<<<M, 1024>>>(a, tok_d);
                        else glue::k_head_local<<<M, 1024>>>(a);
                    }
                    k_loop<<<NR, 128>>>(loop);
                    for (int r = 0; r < NR; ++r)
                    {
                        glue::HeadArgs a = A[r];
                        a.rows = rows_d + (size_t)s * M; a.tok_out = out[r * 2 + k] + (size_t)s * M; a.lp_out = lpo[r * 2 + k] + (size_t)s * M;
                        if (nc > 0 && k) glue::k_head_merge_tp_rs<<<M, 1024>>>(a, tok_d);
                        else if (nc > 0) glue::k_head_merge_tp<<<M, 1024>>>(a);
                        else if (k) glue::k_head_merge_rs<<<1, 32>>>(a, tok_d);
                        else glue::k_head_merge<<<1, 32>>>(a);
                    }
                }
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            unsigned e = 0, mi = 0;
            CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&mi, miss, 4, cudaMemcpyDeviceToHost));
            if (e) { fprintf(stderr, "fabric error word %08x\n", e); return 1; }
            std::vector<std::vector<int>> ho(NR * 2, std::vector<int>((size_t)N * M));
            std::vector<std::vector<float>> hl(NR * 2, std::vector<float>((size_t)N * M));
            for (int i = 0; i < NR * 2; ++i)
            {
                CK(cudaMemcpy(ho[i].data(), out[i], ho[i].size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hl[i].data(), lpo[i], hl[i].size() * 4, cudaMemcpyDeviceToHost));
            }
            printf("== NR %d, T %g, %s, %ld seeds, top_p misses %u\n", NR, T, nc > 0 ? ("top_p " + std::to_string(tp) + ", " + std::to_string(nc) + " candidates per rank").c_str() : "full vocabulary", N, mi);
            for (int r = 1; r < NR; ++r)
                for (int k = 0; k < 2; ++k)
                    if (ho[r * 2 + k] != ho[k]) { printf("FAIL: rank %d writes other tokens than rank 0 (%s)\n", r, k ? "rejection" : "coupled"); fail = true; }
            if (T == 0.f)
            {
                const bool same = ho[0] == ho[1] && memcmp(hl[0].data(), hl[1].data(), hl[0].size() * 4) == 0;
                printf("T = 0: rejection kernels %s the coupled kernels' tokens and logprobs\n", same ? "give" : "DIFFER from");
                if (!same) fail = true;
                continue;
            }
            long d7 = 0;
            for (long s = 0; s < N; ++s)
                d7 += ho[0][(size_t)s * M + 7] != ho[1][(size_t)s * M + 7] || memcmp(&hl[0][(size_t)s * M + 7], &hl[1][(size_t)s * M + 7], 4);
            printf("row 7 (no draft): %ld of %ld seeds differ between the kernels\n", d7, N);
            if (d7) fail = true;
            for (int m = 0; m < M; ++m)
            {
                const std::vector<double> p = target_p(&lg[(size_t)m * V], inv_temp, tp);
                const int d = m < M - 1 ? tok[m + 1] : -1;
                for (int k = 0; k < 2; ++k)
                {
                    std::vector<long> obs(V, 0);
                    long kept = 0;
                    for (long s = 0; s < N; ++s) { const int t = ho[k][(size_t)s * M + m]; ++obs[t]; kept += t == d; }
                    int df;
                    long outside;
                    const double pv = chi2_p(obs, p, N, df, outside);
                    printf("row %d %-9s chi-square p %.4f (df %d)%s", m, k ? "rejection" : "coupled", pv, df, outside ? " OUTSIDE THE NUCLEUS" : "");
                    if (d >= 0)
                    {
                        // the share of seeds that kept the draft against p(d): two-sided binomial z-test (exact when
                        // p(d) is 0 or 1: then every seed must agree)
                        const double pd = p[d], sd = std::sqrt((double)N * pd * (1 - pd));
                        const double kp = sd > 0 ? std::erfc(std::fabs(kept - N * pd) / sd / std::sqrt(2.0)) : (kept == std::lround(N * pd) ? 1.0 : 0.0);
                        printf("  draft kept %.4f, p(draft) %.4f, binomial p %.4f", (double)kept / N, pd, kp);
                        if (kp < 1e-3) fail = true;
                    }
                    printf("\n");
                    if (pv < 1e-3 || outside) fail = true;
                }
            }
        }
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail ? 1 : 0;
}
