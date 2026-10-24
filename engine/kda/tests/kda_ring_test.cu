// GPU check of the KDA carry mode (Args::src, the MTP verify ring) on synthetic inputs. One layer, NH own heads.
//   ring: from state X in slot src, a verify step of M = k + 1 rows with row_slot[r] = r in carry mode; for every
//         accepted count a in 0..k, slot a must be bit-equal to the state (S and conv history) after an M = a + 1 run
//         of rows 0..a from X in the stock mode (src = nullptr, one slot), and the gated outputs of rows 0..a bit-equal.
//         src runs over slots outside and inside the ring (a row then overwrites the source slot in the same step).
//   plain: carry mode with row_slot[*] = src equals the stock mode (state and outputs).
//   sessions: two verify steps in one launch (Args::src per row: each session's first row loads its own source), the
//         second session's slots after the first's; every slot of both equals the ring case's.
//
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -o kda_ring_test kda_ring_test.cu
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "../kda.cuh"

using namespace kda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

constexpr int NH = 2, MXR = 8, SLOTS = 2 * (MXR + 1);
constexpr int DL = NH * DK;

static uint16_t host_bf(float f) { uint32_t u; memcpy(&u, &f, 4); return static_cast<uint16_t>(u >> 16); }
template <class T> static T* up(const std::vector<T>& h)
{
    T* d;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n)
{
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

struct Buf
{
    float *S = nullptr, *cst = nullptr;
    int slots = 0;
    size_t s_slot() const { return (size_t)NH * DK * DV; }
    size_t c_slot() const { return (size_t)NHIST * 3 * NH * DK; }
    void alloc(int n)
    {
        slots = n;
        CK(cudaMalloc(&S, n * s_slot() * 4));
        CK(cudaMalloc(&cst, n * c_slot() * 4));
        CK(cudaMemset(S, 0, n * s_slot() * 4));
        CK(cudaMemset(cst, 0, n * c_slot() * 4));
    }
    void put(int slot, const std::vector<float>& s, const std::vector<float>& c)
    {
        CK(cudaMemcpy(S + slot * s_slot(), s.data(), s_slot() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cst + slot * c_slot(), c.data(), c_slot() * 4, cudaMemcpyHostToDevice));
    }
    std::vector<float> s_of(int slot) const { return down(S + slot * s_slot(), s_slot()); }
    std::vector<float> c_of(int slot) const { return down(cst + slot * c_slot(), c_slot()); }
};

int main()
{
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    auto vec = [&](size_t n, float sc) { std::vector<float> v(n); for (float& x : v) x = sc * U(rng); return v; };
    auto bvec = [&](size_t n, float sc, float off = 0.f) { std::vector<uint16_t> v(n); for (auto& x : v) x = host_bf(off + sc * U(rng)); return v; };

    PLayout pl; pl.dl = DL;
    const int ld = pl.ld();
    const std::vector<float> P = vec((size_t)MXR * ld, 1.f);
    const std::vector<uint16_t> fb = bvec((size_t)DL * DK, 0.05f), gb = bvec((size_t)DL * DK, 0.05f);
    const std::vector<uint16_t> cw0 = bvec((size_t)DL * CONV, 0.5f), cw1 = bvec((size_t)DL * CONV, 0.5f), cw2 = bvec((size_t)DL * CONV, 0.5f);
    const std::vector<float> A_log = vec(NH, 0.5f), dt_bias = vec(DL, 0.5f);
    const std::vector<uint16_t> onorm = bvec(DK, 0.2f, 1.f);
    const std::vector<float> S0 = vec((size_t)NH * DK * DV, 0.1f), C0 = vec((size_t)NHIST * 3 * NH * DK, 1.f);

    Args base;
    base.nh = NH; base.h0 = 0; base.nhmax = NH;
    base.conv_w[0] = up(cw0); base.conv_w[1] = up(cw1); base.conv_w[2] = up(cw2);
    base.fb = up(fb); base.gb = up(gb); base.A_log = up(A_log); base.dt_bias = up(dt_bias); base.onorm = up(onorm);
    base.P = up(P);
    uint16_t* OH;
    CK(cudaMalloc(&OH, (size_t)MXR * DL * 2));
    base.OH = OH;
    int *row_slot, *src;
    CK(cudaMalloc(&row_slot, MXR * 4));
    CK(cudaMalloc(&src, MXR * 4));

    // stock-mode reference: rows 0..n-1 from X in one slot -> state, outputs
    auto stock = [&](int n, std::vector<float>& s, std::vector<float>& c, std::vector<uint16_t>& o) {
        Buf b; b.alloc(1); b.put(0, S0, C0);
        Args a = base; a.S = b.S; a.cst = b.cst; a.M = n; a.src = nullptr;
        CK(cudaMemset(row_slot, 0, MXR * 4));
        a.row_slot = row_slot;
        CK(cudaMemset(OH, 0, (size_t)MXR * DL * 2));
        launch_core(a, 0, false);
        CK(cudaDeviceSynchronize());
        s = b.s_of(0); c = b.c_of(0); o = down(OH, (size_t)n * DL);
        CK(cudaFree(b.S)); CK(cudaFree(b.cst));
    };
    // carry mode: rows 0..n-1 with the given slots, each run from X in its source slot (srcs[m] >= 0 starts a run)
    auto carry_runs = [&](int n, const std::vector<int>& slots, const std::vector<int>& srcs, Buf& b, std::vector<uint16_t>& o,
                          const float* rows = nullptr) {
        b.alloc(SLOTS);
        for (int m = 0; m < n; ++m) if (srcs[m] >= 0) b.put(srcs[m], S0, C0);
        Args a = base; a.S = b.S; a.cst = b.cst; a.M = n; a.src = src;
        if (rows) a.P = rows;
        CK(cudaMemcpy(row_slot, slots.data(), n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(src, srcs.data(), n * 4, cudaMemcpyHostToDevice));
        a.row_slot = row_slot;
        CK(cudaMemset(OH, 0, (size_t)MXR * DL * 2));
        launch_core(a, 0, false);
        CK(cudaDeviceSynchronize());
        o = down(OH, (size_t)n * DL);
    };
    auto carry = [&](int n, const std::vector<int>& slots, int sv, Buf& b, std::vector<uint16_t>& o) {
        std::vector<int> srcs(n, -1);
        srcs[0] = sv;
        carry_runs(n, slots, srcs, b, o);
    };

    std::vector<std::vector<float>> rs(MXR + 1), rc(MXR + 1);
    std::vector<std::vector<uint16_t>> ro(MXR + 1);
    for (int n = 1; n <= MXR; ++n) stock(n, rs[n], rc[n], ro[n]);

    int checks = 0, fails = 0;
    for (int k = 1; k < MXR; ++k)
        for (int sv : {k + 1, 0, k / 2 + 1 <= k ? k / 2 + 1 : k, k})
        {
            std::vector<int> slots(k + 1);
            for (int r = 0; r <= k; ++r) slots[r] = r;
            Buf b;
            std::vector<uint16_t> o;
            carry(k + 1, slots, sv, b, o);
            for (int a = 0; a <= k; ++a)
            {
                const bool okS = b.s_of(a) == rs[a + 1], okC = b.c_of(a) == rc[a + 1];
                const bool okO = memcmp(o.data(), ro[a + 1].data(), (size_t)(a + 1) * DL * 2) == 0;
                ++checks;
                if (!(okS && okC && okO)) { ++fails; printf("FAIL ring k %d src %d a %d: S %d conv %d out %d\n", k, sv, a, okS, okC, okO); }
            }
            CK(cudaFree(b.S)); CK(cudaFree(b.cst));
        }
    for (int n = 1; n <= MXR; ++n)
        for (int sv : {0, 3})
        {
            Buf b;
            std::vector<uint16_t> o;
            carry(n, std::vector<int>(n, sv), sv, b, o);
            const bool ok = b.s_of(sv) == rs[n] && b.c_of(sv) == rc[n] && o == ro[n];
            ++checks;
            if (!ok) { ++fails; printf("FAIL plain n %d src %d\n", n, sv); }
            CK(cudaFree(b.S)); CK(cudaFree(b.cst));
        }
    for (int n : {3, 4, MXR})   // k_kda_core's M <= 4 and general paths
    for (int ka = 1; ka < n - 1; ++ka)
    {
        const int kb = n - 2 - ka, na = ka + 1, base_b = MXR + 1;
        std::vector<int> slots(n), srcs(n, -1);
        for (int r = 0; r < na; ++r) slots[r] = r;
        for (int r = 0; r <= kb; ++r) slots[na + r] = base_b + r;
        srcs[0] = ka; srcs[na] = base_b + kb / 2;
        // the second session's rows are the projections of rows 0..kb, so both compare with the stock runs
        std::vector<float> P2(P.begin(), P.begin() + (size_t)na * ld);
        P2.insert(P2.end(), P.begin(), P.begin() + (size_t)(kb + 1) * ld);
        float* rows = up(P2);
        Buf b;
        std::vector<uint16_t> o;
        carry_runs(na + kb + 1, slots, srcs, b, o, rows);
        CK(cudaFree(rows));
        bool ok = true;
        for (int a = 0; a <= ka; ++a)
            ok = ok && b.s_of(a) == rs[a + 1] && b.c_of(a) == rc[a + 1] && !memcmp(o.data(), ro[a + 1].data(), (size_t)(a + 1) * DL * 2);
        for (int a = 0; a <= kb; ++a)
            ok = ok && b.s_of(base_b + a) == rs[a + 1] && b.c_of(base_b + a) == rc[a + 1] &&
                 !memcmp(o.data() + (size_t)na * DL, ro[a + 1].data(), (size_t)(a + 1) * DL * 2);
        ++checks;
        if (!ok) { ++fails; printf("FAIL sessions n %d k %d + %d\n", n, ka, kb); }
        CK(cudaFree(b.S)); CK(cudaFree(b.cst));
    }
    printf("kda_ring_test: %d checks, %d failed -> %s\n", checks, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
