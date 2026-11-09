// lm_head GEMV at the DeepSeek V4.1 width (K = HID = 5120) on a rank's vocab slice: the real bind::make_gv / bind::launch BF16W
// path that rank.cuh head_for() uses (codec BF16W when G.lm_head is a [O/16][K/16][512] record without scale), fed by the real
// fragment writer (flash_mhc::fragments, same table layout ds_mhc::finish writes), against a CPU dot product (double) of the
// same bf16 operands. Synthetic hash-filled weights in the pack's BF16W record layout (densev.cuh Bf16w): record = one 16x16
// tile = 512 B, lane l (r = l/4, c = l%4) holds a0..a3 = (row r, k 2c..), (row r+8, k 2c..), (row r, k 2c+8..), (row r+8, k 2c+8..).
//   ds_head_gemv_test [tiles ...]      tiles = 128-row vocab tiles of the slice (TP4: 253 / 252, TP6: 169 / 168); default 253 169
// Build (on the Spark, aarch64):  nvcc -O2 -std=c++17 -gencode=arch=compute_121,code=sm_121 -DG6_MODEL_DS41=1 -DDEC_S2 -I engine
//                                      -o ds_head_gemv_test engine/runtime/tests/ds_head_gemv_test.cu
// Needs a GPU (device ~ tiles * 1.3 MB + small, host ~ 2x the slice). Exit 0 all PASS, 1 any FAIL, 2 CUDA error.
#include "../../mhc/ds_mhc.cuh"
#include "../../kernels/bind.cuh"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static uint32_t hash32(uint64_t i, uint32_t seed)
{
    uint64_t z = i * 0x9e3779b97f4a7c15ull + seed;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return static_cast<uint32_t>((z ^ (z >> 31)) >> 16);
}
static uint16_t f2b(float f)       // round-to-nearest-even bf16 of a finite float
{
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
static float b2f(uint16_t b) { uint32_t u = static_cast<uint32_t>(b) << 16; float f; memcpy(&f, &u, 4); return f; }

static int g_pass = 0, g_fail = 0;
static void report(bool ok, const char* fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    printf("%s %s\n", ok ? "PASS" : "FAIL", buf);
    fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

constexpr int MROWS = G6_MX;

static void run_slice(int tiles)
{
    const int K = HID, O = tiles * 128, OT = O / 16, NG = K / 16;
    printf("slice: %d tiles of 128 = %d vocab rows, K = %d, bf16 head %.1f MB\n", tiles, O, K, (double)O * K * 2 / 1e6);

    // --- weights W[O][K] bf16, input x[MROWS][K] bf16 (values kept as floats)
    std::vector<uint16_t> W(static_cast<size_t>(O) * K);
    for (size_t i = 0; i < W.size(); ++i) W[i] = f2b((static_cast<int>(hash32(i, 11) & 0xffff) - 32768) * (0.05f / 32768.f));
    std::vector<float> X(static_cast<size_t>(MROWS) * K);
    for (size_t i = 0; i < X.size(); ++i) X[i] = b2f(f2b((static_cast<int>(hash32(i, 77) & 0xffff) - 32768) * (1.f / 32768.f)));

    // --- BF16W records [OT][NG][512 B]
    uint8_t* w_dev = nullptr;
    {
        std::vector<uint32_t> rec(static_cast<size_t>(OT) * NG * 128);
        for (int ot = 0; ot < OT; ++ot)
            for (int g = 0; g < NG; ++g)
                for (int lane = 0; lane < 32; ++lane)
                {
                    const int r = lane >> 2, c = lane & 3;
                    auto pair = [&](int row, int k) {
                        const uint16_t* p = &W[static_cast<size_t>(16 * ot + row) * K + 16 * g + k];
                        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 16);
                    };
                    uint32_t* q = &rec[(static_cast<size_t>(ot) * NG + g) * 128 + lane * 4];
                    q[0] = pair(r, 2 * c); q[1] = pair(r + 8, 2 * c); q[2] = pair(r, 2 * c + 8); q[3] = pair(r + 8, 2 * c + 8);
                }
        CK(cudaMalloc(&w_dev, rec.size() * 4));
        CK(cudaMemcpy(w_dev, rec.data(), rec.size() * 4, cudaMemcpyHostToDevice));
    }

    // --- CPU reference (double): rows 0..3 at every output, rows 4.. at a stride-7 subset plus the first/last tile
    std::vector<double> ref(static_cast<size_t>(MROWS) * O, NAN), aref(static_cast<size_t>(MROWS) * O, 0.0);
    {
        std::vector<float> wrow(K);
        for (int o = 0; o < O; ++o)
        {
            bool any = false;
            for (int m = 0; m < MROWS; ++m) any |= m < 4 || o % 7 == 0 || o < 128 || o >= O - 128;
            if (!any) continue;
            for (int k = 0; k < K; ++k) wrow[k] = b2f(W[static_cast<size_t>(o) * K + k]);
            for (int m = 0; m < MROWS; ++m)
            {
                if (!(m < 4 || o % 7 == 0 || o < 128 || o >= O - 128)) continue;
                const float* x = &X[static_cast<size_t>(m) * K];
                double s = 0, a = 0;
                for (int k = 0; k < K; ++k) { const double p = static_cast<double>(wrow[k]) * x[k]; s += p; a += fabs(p); }
                ref[static_cast<size_t>(m) * O + o] = s;
                aref[static_cast<size_t>(m) * O + o] = a;
            }
        }
    }
    std::vector<uint16_t>().swap(W);

    // --- device: x as float -> bf16 fragment table (flash_mhc::fragments), logits [MROWS][O] fp32
    float* x_dev; u32* tab; float* y_dev;
    const size_t tbytes = static_cast<size_t>(KSH) * 64 * 4 * g6mx::NT;      // what scratch_layout allocates for xf_head
    CK(cudaMalloc(&x_dev, X.size() * 4)); CK(cudaMemcpy(x_dev, X.data(), X.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&tab, tbytes)); CK(cudaMemset(tab, 0xff, tbytes));
    CK(cudaMalloc(&y_dev, static_cast<size_t>(MROWS) * O * 4));
    bind::Tensor t;
    t.p = w_dev; t.nd = 3; t.shape[0] = OT; t.shape[1] = NG; t.shape[2] = 512;

    struct Cfg { int tpc, ks; };
    const Cfg cfgs[] = {{1, 32}, {1, 16}, {2, 16}, {2, 8}, {4, 8}, {1, 8}};      // {1, 32} is what head_for() uses
    const int Ms[] = {1, 2, 3, 4, 5, 8, 9, 16, 24, 32};
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (const Cfg& cf : cfgs)
    {
        for (int M : Ms)
        {
            if (M > MROWS) continue;
            CK(cudaMemset(y_dev, 0xff, static_cast<size_t>(MROWS) * O * 4));
            CK(cudaMemset(tab, 0xff, tbytes));
            flash_mhc::fragments(x_dev, M, K, nullptr, tab, nullptr);
            bind::Gv g;
            if (const char* e = bind::make_gv(t, bind::BF16W, cf.tpc, cf.ks, M, tab, y_dev, gv::O_F32, g)) { report(false, "make_gv: %s", e); continue; }
            if (!g.on || g.K != K || g.O != O) { report(false, "make_gv: K %d O %d", g.K, g.O); continue; }
            bind::launch(g, nullptr);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<float> y(static_cast<size_t>(MROWS) * O);
            CK(cudaMemcpy(y.data(), y_dev, y.size() * 4, cudaMemcpyDeviceToHost));
            size_t bad = 0, checked = 0, unwritten = 0, nonfinite = 0;
            double worst = 0, maxabs = 0;
            for (int m = 0; m < MROWS; ++m)
                for (int o = 0; o < O; ++o)
                {
                    const float v = y[static_cast<size_t>(m) * O + o];
                    uint32_t u; memcpy(&u, &v, 4);
                    if (m >= M) { unwritten += u != 0xffffffffu; continue; }
                    if (!std::isfinite(v)) { ++nonfinite; continue; }
                    const double r = ref[static_cast<size_t>(m) * O + o];
                    if (std::isnan(r)) continue;
                    ++checked;
                    const double err = fabs(v - r), tol = 5e-5 * aref[static_cast<size_t>(m) * O + o] + 1e-7;
                    maxabs = fmax(maxabs, fabs(r));
                    worst = fmax(worst, err / tol);
                    bad += err > tol;
                }
            const bool ok = !bad && !nonfinite && !unwritten && maxabs > 0.1;
            // time: weights are streamed from DRAM (slice >> L2)
            const int reps = 100;
            bind::launch(g, nullptr);
            CK(cudaEventRecord(e0));
            for (int i = 0; i < reps; ++i) bind::launch(g, nullptr);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / reps;
            report(ok, "tiles %d tpc %d ks %2d M %2d: %zu cells checked, %zu bad, %zu nonfinite, %zu rows>=M written, worst err/tol %.3f, max|ref| %.2f | %.1f us %.0f GB/s",
                   tiles, cf.tpc, cf.ks, M, checked, bad, nonfinite, unwritten, worst, maxabs, us, (double)O * K * 2 / (us * 1e3));
        }
    }
    CK(cudaFree(w_dev)); CK(cudaFree(x_dev)); CK(cudaFree(tab)); CK(cudaFree(y_dev));
}

int main(int argc, char** argv)
{
    std::vector<int> tl;
    for (int i = 1; i < argc; ++i) tl.push_back(atoi(argv[i]));
    if (tl.empty()) tl = {253, 169};
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    printf("ds_head_gemv_test: HID %d KSH %d VOCAB %d, G6_MX %d, device free %.1f GB\n", HID, KSH, geom::VOCAB, G6_MX, fr / 1e9);
    for (int t : tl) run_slice(t);
    printf("%d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
