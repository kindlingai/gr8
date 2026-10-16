// Weight-major MoE bench: bindp::k_pf_fc1ws / k_pf_fc2w (Expert::wm, wm2) vs k_pf_fc1 / k_pf_fc2v (h16 + bs + hx) on random
// K4 words, hx rows, x2 fragments, v1b / v3b scales and router logits: z1 and yw must be byte-equal; times both.
// moe_wm_bench M [REPS] [E4] [SKEW]   SKEW > 0: router logits get a per-expert offset SKEW * N(0, 1) (uneven expert loads)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "kernels/bind_prefill.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

template <class F> float timeit(F f, int reps)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f();
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

__global__ void k_fill_halves(__half* p, size_t n, uint32_t seed)      // finite values in [-2, 2)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        uint32_t x = static_cast<uint32_t>(i) * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13;
        p[i] = __float2half(static_cast<float>(x >> 22) / 256.f - 2.f);
    }
}
__global__ void k_fill_bytes(uint8_t* p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n / 4; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        uint32_t x = static_cast<uint32_t>(i) * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
        reinterpret_cast<uint32_t*>(p)[i] = x;
    }
}

int main(int argc, char** argv)
{
    using namespace bindp;
    constexpr int EB = g6nr::EB, NE = geom::NEXP;
    const int M = argc > 1 ? atoi(argv[1]) : 8192, reps = argc > 2 ? atoi(argv[2]) : 10, e4 = argc > 3 ? atoi(argv[3]) : 1;
    const float skew = argc > 4 ? static_cast<float>(atof(argv[4])) : 0.f;
    const size_t UNIT = static_cast<size_t>(KSH) * 8 * 128;          // bytes of one K4 fc1 matrix (128 cols)
    std::mt19937 g(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    // table: expert e = block rows e * EB .., nb = EB (every 7th expert EB - 2: the b >= nb CTAs), K4, no kmix
    std::vector<long long> tab(static_cast<size_t>(NE) * 6);
    for (int e = 0; e < NE; ++e)
    {
        long long* r = &tab[static_cast<size_t>(e) * 6];
        r[0] = static_cast<long long>(e) * 2 * EB * UNIT; r[1] = static_cast<long long>(e) * EB * UNIT; r[2] = static_cast<long long>(e) * EB;
        r[3] = e % 7 == 3 ? EB - 2 : EB; r[4] = 4; r[5] = static_cast<long long>(e) * EB;
    }
    const size_t nfc1 = static_cast<size_t>(NE) * 2 * EB * UNIT;
    std::vector<__half> hv1b(static_cast<size_t>(NE) * EB * 2 * (HID / 128)), hsg(2 * HID), hhx(static_cast<size_t>(M) * HID);
    for (auto& v : hv1b) v = __float2half(ud(g) * 0.05f);
    for (auto& v : hsg) v = __float2half(nd(g) > 0 ? 1.f : -1.f);
    for (auto& v : hhx) v = __float2half(nd(g));
    std::vector<float> hlog(static_cast<size_t>(M) * NE), hbias(NE, 0.f);
    for (auto& v : hlog) v = nd(g);
    if (skew > 0.f)
    {
        std::vector<float> off(NE);
        for (auto& v : off) v = skew * nd(g);
        for (size_t i = 0; i < hlog.size(); ++i) hlog[i] += off[i % NE];
    }

    long long* d_tab; uint8_t* d_fc1; __half *d_v1b, *d_sg, *d_hx, *z1a, *z1b; float *d_log, *d_bias;
    CK(cudaMalloc(&d_tab, tab.size() * 8)); CK(cudaMalloc(&d_fc1, nfc1));
    CK(cudaMalloc(&d_v1b, hv1b.size() * 2)); CK(cudaMalloc(&d_sg, hsg.size() * 2)); CK(cudaMalloc(&d_hx, hhx.size() * 2));
    CK(cudaMalloc(&d_log, hlog.size() * 4)); CK(cudaMalloc(&d_bias, NE * 4));
    CK(cudaMemcpy(d_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v1b, hv1b.data(), hv1b.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sg, hsg.data(), hsg.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_hx, hhx.data(), hhx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_log, hlog.data(), hlog.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bias, hbias.data(), NE * 4, cudaMemcpyHostToDevice));
    k_fill_bytes<<<1024, 256>>>(d_fc1, nfc1, 0x1234567u);
    const size_t R = static_cast<size_t>(M) * TOPK, nz = R * 2 * EB * 128;
    CK(cudaMalloc(&z1a, nz * 2)); CK(cudaMalloc(&z1b, nz * 2));
    CK(cudaMemset(z1a, 0xff, nz * 2)); CK(cudaMemset(z1b, 0xff, nz * 2));
    int *tok_e, *cnt, *rr_row, *rr_e, *rr_sp, *pos, *segs, *n_seg, *wt, *n_wt;
    float *tok_w, *rr_w;
    CK(cudaMalloc(&tok_e, R * 4)); CK(cudaMalloc(&tok_w, R * 4)); CK(cudaMalloc(&cnt, NE * 4));
    CK(cudaMalloc(&rr_row, R * 4)); CK(cudaMalloc(&rr_e, R * 4)); CK(cudaMalloc(&rr_sp, R * 4)); CK(cudaMalloc(&rr_w, R * 4)); CK(cudaMalloc(&pos, R * 4));
    CK(cudaMalloc(&segs, static_cast<size_t>(max_segments(M)) * 12)); CK(cudaMalloc(&n_seg, 4));
    CK(cudaMalloc(&wt, static_cast<size_t>(max_wtiles(M)) * 12)); CK(cudaMalloc(&n_wt, 4));

    Expert A;
    A.on = 1; A.M = M; A.maxseg = max_segments(M);
    A.h16 = 1; A.e4 = e4; A.fc2one = 1; A.fc2v2 = 1; A.hx = d_hx;
    A.logits = d_log; A.bias = d_bias;
    A.tok_e = tok_e; A.tok_w = tok_w; A.cnt = cnt; A.rr_row = rr_row; A.rr_e = rr_e; A.rr_sp = rr_sp; A.rr_w = rr_w; A.pos = pos;
    A.segs = segs; A.n_seg = n_seg; A.wt = wt; A.n_wt = n_wt;
    A.t_tab = d_tab; A.t_kmix = nullptr; A.t_fc1 = d_fc1; A.k2 = 0; A.bs = 1; A.t_sg = d_sg; A.t_v1b = d_v1b; A.t_wide = 0;
    A.x = d_hx; A.ldx = HID;
    launch_route(A, 0);
    CK(cudaDeviceSynchronize());
    int ns = 0, nw = 0;
    CK(cudaMemcpy(&ns, n_seg, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&nw, n_wt, 4, cudaMemcpyDeviceToHost));
    std::vector<int> hc(NE);
    CK(cudaMemcpy(hc.data(), cnt, NE * 4, cudaMemcpyDeviceToHost));
    int cmin = 1 << 30, cmax = 0;
    for (int c : hc) { cmin = std::min(cmin, c); cmax = std::max(cmax, c); }
    printf("M %d e4 %d EB %d: segments %d wtiles %d, rows per expert %d..%d\n", M, e4, EB, ns, nw, cmin, cmax);

    Expert Ao = A, An = A;
    Ao.wm = 0; Ao.z1 = z1a;
    An.wm = 1; An.z1 = z1b;
    launch_fc1<EB>(Ao, 0);
    launch_fc1<EB>(An, 0);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> a(nz), b(nz);
    CK(cudaMemcpy(a.data(), z1a, nz * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), z1b, nz * 2, cudaMemcpyDeviceToHost));
    size_t diff = 0, written = 0, first = nz;
    for (size_t i = 0; i < nz; ++i)
    {
        if (a[i] != 0xffff) ++written;
        if (a[i] != b[i]) { if (!diff) first = i; ++diff; }
    }
    printf("z1 halves %zu written %zu differ %zu%s\n", nz, written, diff, diff ? "" : "  BYTE-EQUAL");
    if (diff)
    {
        const size_t i = first;
        printf("first diff at %zu (route %zu unit %zu col %zu): old %04x new %04x\n", i, i / (2 * EB * 128), (i / 128) % (2 * EB), i % 128, a[i], b[i]);
    }
    const float to = timeit([&] { launch_fc1<EB>(Ao, 0); }, reps);
    const float tn = timeit([&] { launch_fc1<EB>(An, 0); }, reps);
    CK(cudaDeviceSynchronize());
    printf("fc1 old %.3f ms  new %.3f ms  (x%.2f)\n", to, tn, to / tn);

    // fc2: x2 = [maxseg][MB][EB][4096 B] fp16 A fragments, fc2 words EB * UNIT per expert, v3b per (block row, group)
    const size_t nx2 = static_cast<size_t>(A.maxseg) * MB * EB * 2048, nfc2 = static_cast<size_t>(NE) * EB * UNIT, ny = R * HID;
    std::vector<__half> hv3b(static_cast<size_t>(NE) * EB * (HID / 128));
    for (auto& v : hv3b) v = __float2half(ud(g) * 0.05f);
    __half *d_x2, *d_v3b, *ywa, *ywb; uint8_t* d_fc2;
    CK(cudaMalloc(&d_x2, nx2 * 2)); CK(cudaMalloc(&d_v3b, hv3b.size() * 2)); CK(cudaMalloc(&d_fc2, nfc2));
    CK(cudaMalloc(&ywa, ny * 2)); CK(cudaMalloc(&ywb, ny * 2));
    CK(cudaMemcpy(d_v3b, hv3b.data(), hv3b.size() * 2, cudaMemcpyHostToDevice));
    k_fill_halves<<<1024, 256>>>(d_x2, nx2, 0x7654321u);
    k_fill_bytes<<<1024, 256>>>(d_fc2, nfc2, 0x2345678u);
    CK(cudaMemset(ywa, 0xff, ny * 2)); CK(cudaMemset(ywb, 0xff, ny * 2));
    for (Expert* X : {&Ao, &An})
    {
        X->x2 = reinterpret_cast<uint8_t*>(d_x2); X->t_fc2 = d_fc2; X->t_v3b = d_v3b; X->t_wide2 = 0;
    }
    Ao.wm2 = 0; Ao.yw = ywa;
    An.wm2 = 1; An.yw = ywb;
    launch_fc2<EB>(Ao, 0);
    launch_fc2<EB>(An, 0);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> ya(ny), yb(ny);
    CK(cudaMemcpy(ya.data(), ywa, ny * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(yb.data(), ywb, ny * 2, cudaMemcpyDeviceToHost));
    size_t d2 = 0, w2 = 0, f2 = ny;
    for (size_t i = 0; i < ny; ++i)
    {
        if (ya[i] != 0xffff) ++w2;
        if (ya[i] != yb[i]) { if (!d2) f2 = i; ++d2; }
    }
    printf("yw halves %zu written %zu differ %zu%s\n", ny, w2, d2, d2 ? "" : "  BYTE-EQUAL");
    if (d2) printf("first diff at %zu (route %zu col %zu): old %04x new %04x\n", f2, f2 / HID, f2 % HID, ya[f2], yb[f2]);
    const float t2o = timeit([&] { launch_fc2<EB>(Ao, 0); }, reps);
    const float t2n = timeit([&] { launch_fc2<EB>(An, 0); }, reps);
    CK(cudaDeviceSynchronize());
    printf("fc2 old %.3f ms  new %.3f ms  (x%.2f)\n", t2o, t2n, t2o / t2n);
    return diff || d2 ? 1 : 0;
}
