// GR8_PF_HXF byte test: k_pf_hx + k_pf_hxq (served Q8 path) vs the fused k_pf_hxqf on random rows. Compares hq and hs
// bytes and times both (cudaEvent median).
//   hxf_test [--M 4096] [--iters 50] [--seed 1] [--scale 1.0]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "../bind_prefill.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

template <class F> static float time_ms(F f, int iters, cudaStream_t st)
{
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(); CK(cudaStreamSynchronize(st));
    std::vector<float> t;
    for (int i = 0; i < iters; ++i) { CK(cudaEventRecord(a, st)); f(); CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms); }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char** argv)
{
    int M = 4096, iters = 50; uint64_t seed = 1; float scale = 1.f;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i]; const char* v = argv[i + 1];
        if (a == "--M") M = atoi(v); else if (a == "--iters") iters = atoi(v); else if (a == "--seed") seed = strtoull(v, nullptr, 10);
        else if (a == "--scale") scale = static_cast<float>(atof(v));
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    constexpr int HID = geom::HID;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> hx(static_cast<size_t>(M) * HID), hs(HID);
    // rows with a per-row magnitude spread (some outlier channels), signs x block scale style t_sg
    for (int r = 0; r < M; ++r)
    {
        const float rs = scale * std::exp(0.5f * nd(rng));
        for (int k = 0; k < HID; ++k) { float v = nd(rng) * rs; if ((k * 7 + r) % 509 == 0) v *= 40.f; hx[static_cast<size_t>(r) * HID + k] = __float2half(v); }
    }
    for (int k = 0; k < HID; ++k) hs[k] = __float2half(((rng() & 1) ? 1.f : -1.f) * (0.5f + static_cast<float>(rng() % 1000) / 1000.f));
    __half *dx, *dsg, *dhx; uint8_t *dq1, *ds1, *dq2, *ds2;
    const size_t xb = static_cast<size_t>(M) * HID * 2, qb = static_cast<size_t>(M) * HID, sb = static_cast<size_t>(M) * (HID / 32);
    CK(cudaMalloc(&dx, xb)); CK(cudaMalloc(&dsg, HID * 2)); CK(cudaMalloc(&dhx, xb));
    CK(cudaMalloc(&dq1, qb)); CK(cudaMalloc(&ds1, sb)); CK(cudaMalloc(&dq2, qb)); CK(cudaMalloc(&ds2, sb));
    CK(cudaMemcpy(dx, hx.data(), xb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsg, hs.data(), HID * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dq1, 0x5a, qb)); CK(cudaMemset(ds1, 0x5a, sb)); CK(cudaMemset(dq2, 0xa5, qb)); CK(cudaMemset(ds2, 0xa5, sb));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    bindp::Expert A; A.on = 1; A.M = M; A.x = dx; A.ldx = HID; A.t_sg = dsg; A.hx = dhx; A.q8 = 1;
    const long long warps = static_cast<long long>((M + 7) / 8) * (HID / 128);
    const unsigned g1 = static_cast<unsigned>((warps + 7) / 8), g2 = static_cast<unsigned>((static_cast<long long>(M) * (HID / 32) + 255) / 256);
    bindp::Expert A1 = A; A1.hq = dq1; A1.hs = ds1;
    bindp::Expert A2 = A; A2.hq = dq2; A2.hs = ds2;
    const float t1 = time_ms([&] { bindp::k_pf_hx<<<g1, 256, 0, st>>>(A1); bindp::k_pf_hxq<<<g2, 256, 0, st>>>(A1); }, iters, st);
    const float t2 = time_ms([&] { bindp::k_pf_hxqf<<<g1, 256, 0, st>>>(A2); }, iters, st);
    CK(cudaGetLastError()); CK(cudaStreamSynchronize(st));
    std::vector<uint8_t> q1(qb), q2(qb), s1(sb), s2(sb);
    CK(cudaMemcpy(q1.data(), dq1, qb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(q2.data(), dq2, qb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s1.data(), ds1, sb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s2.data(), ds2, sb, cudaMemcpyDeviceToHost));
    size_t dq = 0, ds = 0; for (size_t i = 0; i < qb; ++i) dq += q1[i] != q2[i]; for (size_t i = 0; i < sb; ++i) ds += s1[i] != s2[i];
    printf("M %d: hx+hxq %.3f ms, hxqf %.3f ms (%.2fx); differing hq bytes %zu of %zu, hs bytes %zu of %zu\n", M, t1, t2, t1 / t2, dq, qb, ds, sb);
    return (dq || ds) ? 1 : 0;
}
