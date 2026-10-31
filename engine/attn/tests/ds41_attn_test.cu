// Test for engine/attn/ds41_attn.cuh against a fixture written by engine/ref/ds41/attn_ref_run.py (the torch twin of the release's
// attention). Every stage is run teacher-forced (fed by the expected output of the stage before it) and compared on its own, then
// layer_forward runs free and its output and intermediates are compared.
//
//   build: nvcc -std=c++17 -O3 -arch=sm_121a -DG6_MODEL_DS41=1 -I engine -o ds41_attn_test engine/attn/tests/ds41_attn_test.cu engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm
//   run:   ds41_attn_test FIXTURE_DIR [--time ITERS]
//          ds41_attn_test --mgate [--M 16,32] [--start ..] [--kind r2,r1,w] [--gv 0|1|both]   (no fixture: chunk == M single-row calls per op; see ds41_attn_mgate.inc)
//          ds41_attn_test --commit [--no-commit]   (no fixture: forced rejection at the compressor, k_comp_commit gate; see commit_test)
//          FIXTURE_DIR written by `attn_ref_run.py decode` (has d.* tensors) also runs the decode-step tests: prefill n0 rows, then D
//          one-row steps (ring R = 128 and R = 136), then the same D rows as one verify chunk at start > 0.
//
// Per stage: eq = fraction of bit-equal elements, ulp1 = fraction within one ulp (bf16 / f32), maxerr = largest |got - expected|,
// maxexp = largest |expected|, rrms = rms(got - expected) / rms(expected). EXACT = bit-equal, GOOD = rrms under the stage's tolerance,
// BAD otherwise. Exit code 1 if any stage is BAD.
#include "../ds41_attn.cuh"
#include "../ds41_attn_gv.cuh"   // single TU: the kernels and the launcher hooks together
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

// The test lives inside namespace ds41a: the GEMV header (densev.cuh -> kernels/common.cuh) declares a global HID, which a using-directive would make ambiguous.
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

// ------------------------------------------------------------------------------------------------ fixture
struct Tensor
{
    std::string dt;
    std::vector<long> shape;
    long numel() const { long n = 1; for (long s : shape) n *= s; return n; }
    size_t esz() const { return dt == "bf16" ? 2 : (dt == "f32" || dt == "i32") ? 4 : 1; }
};

struct Fix
{
    std::string dir;
    int layer = 0, n = 0, D = -1;
    std::map<std::string, Tensor> t;
    bool has(const std::string& k) const { return t.count(k) != 0; }
    const Tensor& info(const std::string& k) const
    {
        auto it = t.find(k);
        if (it == t.end()) { fprintf(stderr, "fixture has no tensor %s\n", k.c_str()); exit(2); }
        return it->second;
    }
    std::vector<uint8_t> raw(const std::string& k) const
    {
        const Tensor& ti = info(k);
        const std::string path = dir + "/" + k + ".bin";
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
        std::vector<uint8_t> v(ti.numel() * ti.esz());
        if (fread(v.data(), 1, v.size(), f) != v.size()) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
        fclose(f);
        return v;
    }
    template <class T> std::vector<T> host(const std::string& k) const
    {
        auto r = raw(k);
        std::vector<T> v(r.size() / sizeof(T));
        memcpy(v.data(), r.data(), r.size());
        return v;
    }
    // an fp8 matrix's scale tensor [N/32][K/32] expanded to the engine's [N][K/32] (each block-row repeated over 32 rows)
    uint8_t* up_scale(const std::string& k) const
    {
        auto r = raw(k);
        const Tensor& ti = info(k);
        const long R = ti.shape[0], C = ti.shape[1];
        std::vector<uint8_t> e((size_t)R * 32 * C);
        for (long i = 0; i < R * 32; i++) memcpy(&e[(size_t)i * C], &r[(size_t)(i / 32) * C], C);
        void* d;
        CK(cudaMalloc(&d, e.size()));
        CK(cudaMemcpy(d, e.data(), e.size(), cudaMemcpyHostToDevice));
        return (uint8_t*)d;
    }
    template <class T> T* up(const std::string& k) const
    {
        auto r = raw(k);
        void* d;
        CK(cudaMalloc(&d, r.size()));
        CK(cudaMemcpy(d, r.data(), r.size(), cudaMemcpyHostToDevice));
        return (T*)d;
    }
    void load(const std::string& d)
    {
        dir = d;
        FILE* f = fopen((d + "/manifest.json").c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s/manifest.json\n", d.c_str()); exit(2); }
        std::string txt;
        char buf[4096];
        size_t k;
        while ((k = fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, k);
        fclose(f);
        auto num_after = [&](const char* key) -> long {
            size_t q = txt.find(key);
            if (q == std::string::npos) return -1;
            q = txt.find(':', q);
            return strtol(txt.c_str() + q + 1, nullptr, 10);
        };
        layer = (int)num_after("\"layer\"");
        n = (int)num_after("\"n\"");
        D = (int)num_after("\"D\"");
        size_t q = txt.find("\"tensors\"");
        q = txt.find('{', q) + 1;
        for (;;)
        {
            while (q < txt.size() && (isspace((unsigned char)txt[q]) || txt[q] == ',')) q++;
            if (q >= txt.size() || txt[q] == '}') break;
            size_t a = txt.find('"', q), b = txt.find('"', a + 1);
            const std::string name = txt.substr(a + 1, b - a - 1);
            size_t d0 = txt.find("\"dtype\"", b);
            size_t s0 = txt.find('"', txt.find(':', d0)), s1 = txt.find('"', s0 + 1);
            Tensor ti;
            ti.dt = txt.substr(s0 + 1, s1 - s0 - 1);
            size_t sh = txt.find("\"shape\"", s1);
            size_t lb = txt.find('[', sh), rb = txt.find(']', lb);
            const char* p = txt.c_str() + lb + 1;
            const char* e = txt.c_str() + rb;
            while (p < e)
            {
                char* nx;
                long v = strtol(p, &nx, 10);
                if (nx == p) { p++; continue; }
                ti.shape.push_back(v);
                p = nx;
            }
            t[name] = ti;
            q = txt.find('}', rb) + 1;
        }
    }
};

// ------------------------------------------------------------------------------------------------ comparison
static int g_bad = 0;
static bool g_time = false;
static int g_iters = 20;
static std::string g_dump;   // --dump DIR: the free-running layer's stage outputs as DIR/L<layer>.<stage>.bin (bf16), the names of the CPU oracle's dumps (engine/ref/ds41/ds41_oracle.py)

struct Stats { long n = 0, nonfinite = 0; double eq = 0, ulp1 = 0, maxerr = 0, maxexp = 0, rrms = 0; };

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static long mono16(uint16_t s) { return (s & 0x8000) ? -(long)(s & 0x7FFF) : (long)(s & 0x7FFF); }
static long mono32(uint32_t s) { return (s & 0x80000000u) ? -(long)(s & 0x7FFFFFFFu) : (long)(s & 0x7FFFFFFFu); }

static Stats stats_bf16(const uint16_t* g, const uint16_t* e, long n)
{
    Stats s;
    s.n = n;
    long eq = 0, u1 = 0;
    double se = 0, sd = 0;
    for (long i = 0; i < n; i++)
    {
        const float gv = bf2f(g[i]), ev = bf2f(e[i]);
        if (!std::isfinite(gv)) { s.nonfinite++; continue; }
        if (g[i] == e[i]) eq++;
        if (labs(mono16(g[i]) - mono16(e[i])) <= 1) u1++;
        const double d = (double)gv - (double)ev;
        s.maxerr = std::max(s.maxerr, fabs(d));
        s.maxexp = std::max(s.maxexp, (double)fabsf(ev));
        sd += d * d;
        se += (double)ev * ev;
    }
    s.eq = (double)eq / n;
    s.ulp1 = (double)u1 / n;
    s.rrms = se > 0 ? sqrt(sd / se) : sqrt(sd);
    return s;
}
static Stats stats_f32(const float* g, const float* e, long n)
{
    Stats s;
    s.n = n;
    long eq = 0, u1 = 0;
    double se = 0, sd = 0;
    for (long i = 0; i < n; i++)
    {
        if (!std::isfinite(g[i])) { s.nonfinite++; continue; }
        uint32_t gu, eu;
        memcpy(&gu, g + i, 4);
        memcpy(&eu, e + i, 4);
        if (gu == eu) eq++;
        if (labs(mono32(gu) - mono32(eu)) <= 1) u1++;
        const double d = (double)g[i] - (double)e[i];
        s.maxerr = std::max(s.maxerr, fabs(d));
        s.maxexp = std::max(s.maxexp, (double)fabsf(e[i]));
        sd += d * d;
        se += (double)e[i] * e[i];
    }
    s.eq = (double)eq / n;
    s.ulp1 = (double)u1 / n;
    s.rrms = se > 0 ? sqrt(sd / se) : sqrt(sd);
    return s;
}
static void report(const char* nm, const Stats& s, double tol)
{
    const char* st = (s.nonfinite == 0 && s.eq == 1.0) ? "EXACT" : (s.nonfinite == 0 && s.rrms < tol) ? "GOOD" : "BAD";
    if (!strcmp(st, "BAD")) g_bad++;
    printf("%-30s %-5s n=%-9ld eq=%.5f ulp1=%.5f maxerr=%.3e maxexp=%.3e rrms=%.3e%s\n", nm, st, s.n, s.eq, s.ulp1, s.maxerr, s.maxexp,
           s.rrms, s.nonfinite ? " NONFINITE!" : "");
}
// compare a device bf16 buffer to host expected values
static Stats cmp_bf16(const char* nm, const void* dptr, const uint16_t* exp, long n, double tol = 3e-3)
{
    std::vector<uint16_t> g(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(g.data(), dptr, n * 2, cudaMemcpyDeviceToHost));
    Stats s = stats_bf16(g.data(), exp, n);
    report(nm, s, tol);
    return s;
}
static Stats cmp_f32(const char* nm, const void* dptr, const float* exp, long n, double tol = 1e-5)
{
    std::vector<float> g(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(g.data(), dptr, n * 4, cudaMemcpyDeviceToHost));
    Stats s = stats_f32(g.data(), exp, n);
    report(nm, s, tol);
    return s;
}

// run fn once (result is what gets compared), then with --time average g_iters runs between events
template <class F> static void timed(const char* nm, F fn)
{
    fn();
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    if (!g_time) return;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < g_iters; i++) fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    printf("    time %-26s %.4f ms\n", nm, ms / g_iters);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

static void dump_bf16(int layer, const char* stage, const void* dev, long count)
{
    if (g_dump.empty()) return;
    std::vector<uint16_t> h((size_t)count);
    CK(cudaMemcpy(h.data(), dev, (size_t)count * 2, cudaMemcpyDeviceToHost));
    char nm[64];
    snprintf(nm, sizeof nm, "/L%d.%s.bin", layer, stage);
    FILE* f = fopen((g_dump + nm).c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s%s\n", g_dump.c_str(), nm); g_bad++; return; }
    fwrite(h.data(), 2, h.size(), f);
    fclose(f);
}

template <class T> static T* dalloc(size_t n)
{
    void* d;
    CK(cudaMalloc(&d, n * sizeof(T)));
    CK(cudaMemset(d, 0, n * sizeof(T)));
    return (T*)d;
}
template <class T> static T* dcopy(const std::vector<T>& v)
{
    T* d = dalloc<T>(v.size());
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static bf16* dbf(const std::vector<uint16_t>& v) { return (bf16*)dcopy(v); }

// ------------------------------------------------------------------------------------------------ decode GEMV (--gv)
static bool g_gv = false;   // --gv: build GEMV records for the fixture's weights, run the GEMV section, and run the decode steps through the GEMV

// MXFP8 decode record of codes [16T][K] + E8M0 scales [16T][K/32] -> [T][K/32][528]: a copy of k_kda_mx_record (engine/kda/kda.cuh), which
// is the loader's records::mxfp8_record (the layout the pack's .mx.gv tensors have). One CTA of 256 threads per record.
__global__ void __launch_bounds__(256) k_mx_record(const uint8_t* codes, const uint8_t* scales, int K, uint8_t* out)
{
    const int ng = K / 32, t = blockIdx.x / ng, g = blockIdx.x % ng, i = threadIdx.x;
    uint8_t* rec = out + (size_t)blockIdx.x * 528;
    const int l = i >> 3, e = i & 7, r = l >> 2, c = l & 3;
    const int row = r + ((e & 2) ? 8 : 0), col = 2 * c + (e & 1) + ((e & 4) ? 8 : 0);
    for (int h = 0; h < 2; ++h) rec[h * 256 + i] = codes[(size_t)(16 * t + row) * K + 32 * g + 16 * h + col];
    if (i < 8)
    {
        rec[512 + 2 * i] = scales[(size_t)(16 * t + i) * ng + g];
        rec[512 + 2 * i + 1] = scales[(size_t)(16 * t + i + 8) * ng + g];
    }
}
static const uint8_t* make_records(const uint8_t* codes, const uint8_t* scales, long rows, int K)
{
    const long T = rows / 16;
    uint8_t* out = dalloc<uint8_t>((size_t)T * (K / 32) * 528);
    k_mx_record<<<(unsigned)(T * (K / 32)), 256>>>(codes, scales, K, out);
    CK(cudaGetLastError());
    return out;
}
// the ceiling: a plain streaming read of n 16-byte words (grid-stride, 4 loads in flight per thread), no math
__global__ void __launch_bounds__(256) k_stream_read(const uint4* __restrict__ p, size_t n, unsigned* sink)
{
    unsigned acc = 0;
    const size_t stride = (size_t)gridDim.x * 256;
    size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride)
    {
        uint32_t a[4], b[4], c[4], d[4];
        gv::ld_nc_v4(p + i, a);
        gv::ld_nc_v4(p + i + stride, b);
        gv::ld_nc_v4(p + i + 2 * stride, c);
        gv::ld_nc_v4(p + i + 3 * stride, d);
        acc ^= a[0] ^ a[1] ^ a[2] ^ a[3] ^ b[0] ^ b[1] ^ b[2] ^ b[3] ^ c[0] ^ c[1] ^ c[2] ^ c[3] ^ d[0] ^ d[1] ^ d[2] ^ d[3];
    }
    for (; i < n; i += stride)
    {
        uint32_t a[4];
        gv::ld_nc_v4(p + i, a);
        acc ^= a[0] ^ a[1] ^ a[2] ^ a[3];
    }
    if (acc == 0x9e3779b9u && threadIdx.x == 255) *sink = acc;   // keeps the loads alive
}

struct ProjCase { const char* name; Fp8Mat W; int groups; };

static std::vector<uint16_t> rand_acts(int M, int K, unsigned seed)
{
    std::vector<uint16_t> h((size_t)M * K);
    uint32_t st = seed * 2654435761u + 12345u;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) * (1.0f / 16777216.0f); };
    for (int m = 0; m < M; m++)
        for (int b = 0; b < K / 32; b++)
        {
            const float sc = ldexpf(1.0f, (int)(rnd() * 8) - 4);   // block magnitudes 2^-4 .. 2^3, as normed activations vary
            for (int j = 0; j < 32; j++)
            {
                float v = (rnd() + rnd() + rnd() - 1.5f) * 2.0f * sc;   // roughly gaussian
                if (rnd() < 0.003f) v *= 12.0f;                         // outliers
                uint32_t u;
                memcpy(&u, &v, 4);
                h[(size_t)m * K + b * 32 + j] = (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
            }
        }
    return h;
}

// GEMV vs the generic GEMM on the same act_quant output; also row independence (M = 1 equals row 0 of M = 6 bit for bit)
static void gv_check(const ProjCase& pc)
{
    const Fp8Mat& W = pc.W;
    const int G = pc.groups, Kt = G * W.K, ldy = G * W.N;
    cudaStream_t st = 0;
    std::vector<uint16_t> keep1;
    for (int M : {1, 6, 8})
    {
        auto hx = rand_acts(M, Kt, 7 + M);
        bf16* x = dbf(hx);
        uint8_t* codes = dalloc<uint8_t>((size_t)M * Kt);
        uint8_t* sc = dalloc<uint8_t>((size_t)M * Kt / 32);
        uint32_t* xf = dalloc<uint32_t>((size_t)Kt * 4);
        bf16* yg = dalloc<bf16>((size_t)M * ldy);
        bf16* yv = dalloc<bf16>((size_t)M * ldy);
        launch_act_quant(x, Kt, codes, Kt, sc, Kt / 32, M, Kt, st);
        launch_gemm(codes, Kt, sc, Kt / 32, W, yg, ldy, M, G, W.K, W.K / 32, W.N, st);
        launch_xfrag(codes, Kt, sc, Kt / 32, M, W.K, G, W.K, W.K / 32, xf, st);
        launch_gemv(xf, W, yv, ldy, M, G, W.N, st);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        std::vector<uint16_t> a((size_t)M * ldy), b((size_t)M * ldy);
        CK(cudaMemcpy(a.data(), yg, a.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), yv, b.size() * 2, cudaMemcpyDeviceToHost));
        const Stats s = stats_bf16(b.data(), a.data(), (long)a.size());
        char nm[96];
        snprintf(nm, sizeof nm, "gemv %s M=%d vs gemm", pc.name, M);
        report(nm, s, 1e-3);
        if (s.ulp1 < 0.999) { printf("    BAD: %.4f of the outputs are more than 1 bf16 ulp from the generic GEMM\n", 1.0 - s.ulp1); g_bad++; }
        if (M == 1) keep1 = b;
        if (M == 6)
        {
            long ne = 0;
            for (int i = 0; i < ldy; i++) ne += keep1[i] != b[i];
            printf("%-30s %s (%ld of %d differ)\n", "  row 0 of M=6 vs M=1", ne ? "BAD" : "EXACT", ne, ldy);
            g_bad += ne != 0;
        }
        for (void* p : {(void*)x, (void*)codes, (void*)sc, (void*)xf, (void*)yg, (void*)yv}) cudaFree(p);
    }
}

static double med_us(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
// median microseconds per launch of fn(i): `steps` back-to-back launches per rep, `reps` reps
template <class F> static double time_us(F fn, int steps, int reps = 7)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < steps; i++) fn(i);   // warm
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int r = 0; r < reps; r++)
    {
        cudaEventRecord(a);
        for (int i = 0; i < steps; i++) fn(i);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        t.push_back(ms * 1000.0 / steps);
    }
    CK(cudaGetLastError());
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return med_us(t);
}

// Per-projection timing: the generic GEMM beside the GEMV at M rows, over a pool of weight copies larger than the L2 (each launch reads a
// different copy, so the weights always come from DRAM), and the streaming-read ceiling over the same bytes.
static void gv_time(const ProjCase& pc, const std::vector<std::pair<int, int>>& cfgs, double& ceil_gbps, unsigned* sink)
{
    const Fp8Mat& W = pc.W;
    const int G = pc.groups, Kt = G * W.K, ldy = G * W.N;
    const long rows = (long)G * W.N;
    const size_t gvb = (size_t)(rows / 16) * (W.K / 32) * 528, cb = (size_t)rows * W.K, sb = (size_t)rows * (W.K / 32);
    const size_t target = 96ull << 20;   // 4x the GB10 L2
    const int P = (int)std::max<size_t>(4, (target + gvb - 1) / gvb);
    uint8_t* pg = dalloc<uint8_t>(P * gvb);
    uint8_t* pc_ = dalloc<uint8_t>(P * cb);
    uint8_t* ps = dalloc<uint8_t>(P * sb);
    for (int i = 0; i < P; i++)
    {
        CK(cudaMemcpy(pg + i * gvb, W.gv, gvb, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(pc_ + i * cb, W.w, cb, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(ps + i * sb, W.s, sb, cudaMemcpyDeviceToDevice));
    }
    auto Wi = [&](int i) { Fp8Mat w = W; w.gv = pg + (size_t)(i % P) * gvb; w.w = pc_ + (size_t)(i % P) * cb; w.s = ps + (size_t)(i % P) * sb; return w; };
    // the ceiling: the whole GEMV pool in one launch (sustained GB/s), and one matrix-sized slice per launch rotating through the pool (the floor
    // a read of this size can reach, launch and ramp included)
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    const unsigned grid = (unsigned)pr.multiProcessorCount * 8;
    const size_t n16 = P * gvb / 16;
    const double pool_us = time_us([&](int) { k_stream_read<<<grid, 256>>>((const uint4*)pg, n16, sink); }, 3, 5);
    const double ceil = (double)P * gvb / 1e3 / pool_us;   // GB/s (bytes / us / 1e3)
    const double slice_us = time_us([&](int i) { k_stream_read<<<grid, 256>>>((const uint4*)(pg + (size_t)(i % P) * gvb), gvb / 16, sink); }, 2 * P);
    ceil_gbps = ceil;
    printf("GV time %-6s | %.2f MB of records (%.2f MB generic codes+scales), pool %d copies | stream ceiling %.1f GB/s (pool) | same-size read %.1f us (%.1f GB/s)\n",
           pc.name, gvb / 1e6, (cb + sb) / 1e6, P, ceil, slice_us, gvb / 1e3 / slice_us);
    for (int M : {1, 6})
    {
        auto hx = rand_acts(M, Kt, 3);
        bf16* x = dbf(hx);
        uint8_t* codes = dalloc<uint8_t>((size_t)M * Kt);
        uint8_t* sc = dalloc<uint8_t>((size_t)M * Kt / 32);
        uint32_t* xf = dalloc<uint32_t>((size_t)Kt * 4);
        bf16* y = dalloc<bf16>((size_t)M * ldy);
        launch_act_quant(x, Kt, codes, Kt, sc, Kt / 32, M, Kt, 0);
        launch_xfrag(codes, Kt, sc, Kt / 32, M, W.K, G, W.K, W.K / 32, xf, 0);
        CK(cudaDeviceSynchronize());
        const int steps = 2 * P;
        const double t_gen = time_us([&](int i) { launch_gemm(codes, Kt, sc, Kt / 32, Wi(i), y, ldy, M, G, W.K, W.K / 32, W.N, 0); }, steps);
        const double t_xf = time_us([&](int) { launch_xfrag(codes, Kt, sc, Kt / 32, M, W.K, G, W.K, W.K / 32, xf, 0); }, steps);
        double best = 1e30;
        int bt = 0, bk = 0;
        std::vector<std::pair<int, int>> all = cfgs;
        all.insert(all.begin(), {0, 0});   // the default pick first
        for (auto c : all)
        {
            const int ks = c.second ? c.second : (g_gv_ks ? g_gv_ks : gv_pick_ks(W.K));
            if ((W.K / 32) % ks || (c.first && c.first * ks > 16)) continue;
            const double t = time_us([&](int i) { launch_gemv(xf, Wi(i), y, ldy, M, G, W.N, 0, c.first, c.second); }, steps);
            const int tpc = c.first ? c.first : g_gv_tpc;
            printf("    M=%d gemv kernel tpc=%d ks=%-2d%s %8.2f us  %6.1f GB/s  %5.1f%% of ceiling\n", M, tpc, ks, c.first ? "  " : " *", t, gvb / 1e3 / t,
                   100.0 * (gvb / 1e3 / t) / ceil);
            if (t < best) { best = t; bt = tpc; bk = ks; }
        }
        printf("  GV time %-6s M=%d | generic gemm %8.2f us | gemv best %8.2f us (tpc=%d ks=%d) %6.1f GB/s = %5.1f%% of the %.1f GB/s ceiling | +xfrag %6.2f us "
               "| speedup %.1fx (%.1fx with xfrag)\n",
               pc.name, M, t_gen, best, bt, bk, gvb / 1e3 / best, 100.0 * (gvb / 1e3 / best) / ceil, ceil, t_xf, t_gen / best, t_gen / (best + t_xf));
        for (void* p : {(void*)x, (void*)codes, (void*)sc, (void*)xf, (void*)y}) cudaFree(p);
    }
    for (void* p : {(void*)pg, (void*)pc_, (void*)ps}) cudaFree(p);
}

static void gv_section(const Weights& W, int G)
{
    printf("---- decode GEMV vs generic GEMM (M = 1, 6, 8)\n");
    const ProjCase cases[] = {{"q_a", W.wq_a, 1}, {"q_b", W.wq_b, 1}, {"kv", W.wkv, 1}, {"o_a", W.wo_a, G}, {"o_b", W.wo_b, 1}};
    for (const ProjCase& pc : cases) gv_check(pc);
    if (!g_time) return;
    printf("---- decode GEMV timing (single projection, weights from DRAM; ms = us / 1000)\n");
    unsigned* sink = dalloc<unsigned>(1);
    const std::vector<std::pair<int, int>> cfgs = {{1, 4}, {1, 8}, {1, 16}, {2, 4}, {2, 8}, {4, 2}, {4, 4}};
    double ceil = 0;
    for (const ProjCase& pc : cases) gv_time(pc, cfgs, ceil, sink);
}

// ------------------------------------------------------------------------------------------------ --commit (forced rejection)
// ds41_attn_test --commit [--no-commit]: no fixture, synthetic weights. The compressor of a ratio-2 source layer (ds41a::comp_row, the
// device-start form the engine runs) fed a DSpark/MTP verify chunk that is partly rejected, then the real tokens one at a time, against the
// same accepted tokens fed one at a time from the start (the n == 1 decode form). After each accepted prefix the compressed rows C.comp[g]
// with 2g + 1 <= last position must match. A case is a prefill of P0 tokens, a verify chunk of n = k + 1 rows at start P0 of which the first
// a + 1 rows are the real tokens (Pa = P0 + a is the last accepted position) and the other k - a rows are DIFFERENT, rejected tokens, then
// ds41a::comp_commit (k_comp_commit) when selected, then the real tokens Pa + 1, Pa + 2, Pa + 3, one row each.
//   gate     : with the commit, every row g with 2g + 1 <= last position is checked after the token Pa + 1 and again after Pa + 3.
//   control  : the same case WITHOUT the commit. The fault is expected exactly when Pa is even and a < k: the state slot 0 then holds the
//              rejected row (or a stale one), so the row completed by token Pa + 1, g = (Pa + 1) / 2, must come out BAD. If the control does not
//              see every expected fault, the test cannot see the bug and FAILS.
//   --no-commit : the gate itself skips the commit call: the run must exit 1 with BAD rows (this is the "commit call removed" check).
// Row verdict: EXACT = all bf16 bits equal, GOOD = rrms < 0.05 (the batch of rows in one f32 gemm may round differently from a single row),
// BAD otherwise (a wrong pair gives rrms ~ 1). Device memory: ~ 2 weight matrices of 5.2 MB + 3 small scratches (< 60 MB). Exit 0 = all ok.
constexpr int CC_MX = 8, CC_HEADS = 16, CC_ROWS = 32;
static uint16_t cc_f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
static void cc_fill(std::vector<uint16_t>& y, size_t off, int len, uint64_t seed, float amp)   // len values in [-amp, amp)
{
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull;
    if (!s) s = 1;
    for (int i = 0; i < len; i++)
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        y[off + i] = cc_f2bf(amp * ((float)((s >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f));
    }
}
struct CcSide { Cache C{}; Scratch S{}; };
static CcSide cc_make()
{
    CcSide s;
    s.C.comp = dalloc<bf16>((size_t)CC_ROWS * HD);
    s.C.kvs = dalloc<float>(2 * HD);
    s.C.scs = dalloc<float>(2 * HD);
    void* base;
    CK(cudaMalloc(&base, Scratch::bytes(CC_MX, CC_HEADS)));
    CK(cudaMemset(base, 0, Scratch::bytes(CC_MX, CC_HEADS)));
    s.S.carve(base, CC_MX, CC_HEADS);
    return s;
}
static void cc_reset(CcSide& s)
{
    CK(cudaMemset(s.C.comp, 0, (size_t)CC_ROWS * HD * 2));
    CK(cudaMemset(s.C.kvs, 0, 2 * HD * sizeof(float)));
    CK(cudaMemset(s.C.scs, 0, 2 * HD * sizeof(float)));
}
// n rows of y at positions start .. start + n - 1 through the engine's compressor step (dev-start form)
static void cc_step(const Weights& W, CcSide& s, const std::vector<uint16_t>& y, int n, int start, bf16* d_y, int* d_start)
{
    CK(cudaMemcpy(d_y, y.data(), (size_t)n * HID * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_start, &start, sizeof(int), cudaMemcpyHostToDevice));
    comp_row(W, s.C, s.S, d_y, n, start, (cudaStream_t)0, d_start);
    CK(cudaGetLastError());
}
struct CcCmp { int rows = 0, exact = 0; std::vector<int> bad; };
static CcCmp cc_cmp(const CcSide& e, const CcSide& t, int rows)
{
    std::vector<uint16_t> he((size_t)rows * HD), ht((size_t)rows * HD);
    CK(cudaMemcpy(he.data(), e.C.comp, he.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ht.data(), t.C.comp, ht.size() * 2, cudaMemcpyDeviceToHost));
    CcCmp r;
    r.rows = rows;
    for (int g = 0; g < rows; g++)
    {
        double d2 = 0, t2 = 0;
        bool eq = true;
        for (int c = 0; c < HD; c++)
        {
            const uint16_t a = he[(size_t)g * HD + c], b = ht[(size_t)g * HD + c];
            eq = eq && a == b;
            const double x = bf2f(a), y = bf2f(b);
            d2 += (x - y) * (x - y);
            t2 += y * y;
        }
        const double rrms = t2 > 0 ? sqrt(d2 / t2) : (d2 > 0 ? 1e9 : 0.0);
        if (eq) r.exact++;
        else if (!(rrms < 0.05)) r.bad.push_back(g);   // also catches NaN
    }
    return r;
}
static std::string cc_list(const std::vector<int>& v)
{
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

static int commit_test(int argc, char** argv)
{
    bool nocommit = false;
    for (int i = 2; i < argc; i++)
        if (!strcmp(argv[i], "--no-commit")) nocommit = true;
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("device %s sm_%d%d | --commit: forced rejection at the ratio-2 compressor%s\n", pr.name, pr.major, pr.minor,
           nocommit ? " | --no-commit: the gate skips k_comp_commit, this run MUST fail" : "");
    std::vector<uint16_t> hk((size_t)HD * HID), hg((size_t)HD * HID), hn(HD);
    cc_fill(hk, 0, (int)hk.size(), 11, 0.03f);
    cc_fill(hg, 0, (int)hg.size(), 12, 0.03f);
    for (int c = 0; c < HD; c++) hn[c] = cc_f2bf(0.5f + (float)(c % 7) * 0.25f);
    float inv[32];
    inv_freqs_host(160000.0, 65536, inv);
    Weights W{};
    W.c_wkv = dbf(hk);
    W.c_wgate = dbf(hg);
    W.c_norm = dbf(hn);
    W.inv = dcopy(std::vector<float>(inv, inv + 32));
    CcSide T = cc_make(), A = cc_make(), B = cc_make();
    bf16* d_y = dalloc<bf16>((size_t)CC_MX * HID);
    int* d_start = dalloc<int>(1);
    int* d_dq = dalloc<int>(1);
    int* d_dacc = dalloc<int>(1);

    int cases = 0, gate_bad = 0, ctl_pred = 0, ctl_hit = 0, ctl_extra = 0;
    for (int P0 = 4; P0 <= 5; P0++)                        // chunk start parity: even, odd
        for (int k = 1; k <= 5; k++)
            for (int a = 0; a <= k; a++)
            {
                const int n = k + 1, p = P0, Pa = p + a, last = Pa + 3;
                const bool predicted = (Pa % 2 == 0) && a < k;     // slot 0 holds the rejected / stale row and the next token starts odd
                const uint64_t tag = 100 + (uint64_t)cases;
                auto real = [&](int pos0, int cnt)
                {
                    std::vector<uint16_t> y((size_t)cnt * HID);
                    for (int r = 0; r < cnt; r++) cc_fill(y, (size_t)r * HID, HID, 1000 + pos0 + r, 1.0f);
                    return y;
                };
                // truth: every accepted token one row at a time
                cc_reset(T);
                for (int pos = 0; pos <= last; pos++) cc_step(W, T, real(pos, 1), 1, pos, d_y, d_start);
                CcCmp res[2][2];   // [variant: 0 gate, 1 control][check: 0 after Pa + 1, 1 after Pa + 3]
                for (int v = 0; v < 2; v++)
                {
                    CcSide& E = v == 0 ? A : B;
                    cc_reset(E);
                    cc_step(W, E, real(0, P0), P0, 0, d_y, d_start);   // prefill
                    std::vector<uint16_t> ch((size_t)n * HID);
                    for (int r = 0; r < n; r++) cc_fill(ch, (size_t)r * HID, HID, r <= a ? 1000 + p + r : 7000 + tag * 16 + r, 1.0f);   // rows past a: rejected tokens
                    cc_step(W, E, ch, n, p, d_y, d_start);
                    if (v == 0 && !nocommit)
                    {
                        CK(cudaMemcpy(d_dq, &Pa, sizeof(int), cudaMemcpyHostToDevice));
                        CK(cudaMemcpy(d_dacc, &a, sizeof(int), cudaMemcpyHostToDevice));
                        comp_commit(E.C, E.S, d_dq, d_dacc, (cudaStream_t)0);
                        CK(cudaGetLastError());
                    }
                    cc_step(W, E, real(Pa + 1, 1), 1, Pa + 1, d_y, d_start);
                    res[v][0] = cc_cmp(E, T, (Pa + 2) / 2);            // rows g with 2g + 1 <= Pa + 1
                    cc_step(W, E, real(Pa + 2, 1), 1, Pa + 2, d_y, d_start);
                    cc_step(W, E, real(Pa + 3, 1), 1, Pa + 3, d_y, d_start);
                    res[v][1] = cc_cmp(E, T, (Pa + 4) / 2);            // rows g with 2g + 1 <= Pa + 3
                }
                const int key = (Pa + 1) / 2;   // the row token Pa + 1 completes when Pa is even
                const bool gate_ok = res[0][0].bad.empty() && res[0][1].bad.empty();
                gate_bad += !gate_ok;
                bool hit = false;
                for (int g : res[1][0].bad) hit = hit || g == key;
                if (predicted) { ctl_pred++; ctl_hit += hit; }
                else ctl_extra += !res[1][0].bad.empty() || !res[1][1].bad.empty();
                printf("P0=%d k=%d n=%d a=%d Pa=%d fault_expected=%d | %s: rows %d/%d exact, bad %s | no commit: bad %s%s\n", P0, k, n, a, Pa, (int)predicted,
                       nocommit ? "NO-COMMIT" : "commit", res[0][1].exact, res[0][1].rows, cc_list(res[0][1].bad).c_str(), cc_list(res[1][1].bad).c_str(),
                       predicted ? (hit ? " (fault seen at the key row, as expected)" : " (FAULT NOT SEEN)") : "");
                cases++;
            }
    printf("cases %d | gate: %d BAD | control: fault expected in %d cases, seen at the key row in %d, unexpected bad rows without commit in %d other cases\n",
           cases, gate_bad, ctl_pred, ctl_hit, ctl_extra);
    int fail = 0;
    if (gate_bad) { printf("FAIL gate: %d case(s) have a compressed row that differs from the one-token-at-a-time truth\n", gate_bad); fail = 1; }
    if (ctl_pred == 0 || ctl_hit != ctl_pred) { printf("FAIL control: the no-commit run did not show the fault in all %d cases where it is expected (seen %d): this test cannot see the bug\n", ctl_pred, ctl_hit); fail = 1; }
    if (!fail) printf("PASS (commit: all rows match the truth; no commit: the fault shows in all %d expected cases)\n", ctl_pred);
    return fail;
}

// GR8_TEST_DS41_MGATE_DIAG, read here because tools/env_lint.py does not scan .inc files
static bool mgate_diag() { return knob::test_ds41_mgate_diag(); }
#include "ds41_attn_mgate.inc"   // --mgate: chunk of M rows == M single-row calls, per op (synthetic weights)

int test_main(int argc, char** argv)
{
    if (argc >= 2 && !strcmp(argv[1], "--commit")) return commit_test(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "--mgate")) return mgate_test(argc, argv);
    if (argc < 2) { fprintf(stderr, "usage: %s FIXTURE_DIR [--time ITERS] [--dump DIR] [--gv] | --commit [--no-commit] | --mgate [--M a,b] [--start a,b] [--kind r2,r1,w] [--gv 0|1|both] [--tk N]\n", argv[0]); return 2; }
    Fix F;
    F.load(argv[1]);
    for (int i = 2; i < argc; i++)
        if (!strcmp(argv[i], "--dump") && i + 1 < argc) g_dump = argv[++i];
        else if (!strcmp(argv[i], "--gv")) g_gv = true;
        else if (!strcmp(argv[i], "--time")) { g_time = true; if (i + 1 < argc && argv[i + 1][0] != '-') g_iters = atoi(argv[++i]); }

    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    const int n = F.n, layer = F.layer;
    const bool comp = F.has("w.compressor.wkv.weight");
    const int heads = (int)F.info("exp.q").shape[1];
    const int G = heads / HPG;
    const int WL = n < WIN ? n : WIN;
    const int TK = comp ? (int)F.info("exp.ids").shape[1] - WL : 0;
    printf("device %s sm_%d%d | fixture %s layer=%d n=%d heads=%d WL=%d TK=%d (%s)\n", pr.name, pr.major, pr.minor, argv[1], layer, n, heads, WL, TK,
           comp ? "ratio-2 compressed" : "window only");
    cudaStream_t st = 0;

    // ---- rope table: the fixture's against the host helper
    {
        float h[32];
        if (layer < 2) inv_freqs_host(10000.0, 0, h);
        else inv_freqs_host(160000.0, 65536, h);
        auto fi = F.host<float>("inv");
        int bad = 0;
        for (int i = 0; i < 32; i++) bad += memcmp(&h[i], &fi[i], 4) != 0;
        printf("%-30s %s (%d of 32 differ)\n", "inv_freqs_host vs fixture", bad ? "BAD" : "EXACT", bad);
        g_bad += bad != 0;
    }

    // ---- weights
    Weights W{};
    W.wq_a = {F.up<uint8_t>("w.wq_a.weight"), F.up_scale("w.wq_a.scale"), QA, HID};
    W.wq_b = {F.up<uint8_t>("w.wq_b.weight"), F.up_scale("w.wq_b.scale"), heads * HD, QA};
    W.wkv = {F.up<uint8_t>("w.wkv.weight"), F.up_scale("w.wkv.scale"), HD, HID};
    W.wo_a = {F.up<uint8_t>("w.wo_a.weight"), F.up_scale("w.wo_a.scale"), OLORA, GROUP_IN};   // N = one group's rows
    W.wo_b = {F.up<uint8_t>("w.wo_b.weight"), F.up_scale("w.wo_b.scale"), HID, G * OLORA};
    W.q_norm = F.up<bf16>("w.q_norm.weight");
    W.kv_norm = F.up<bf16>("w.kv_norm.weight");
    W.sink = F.up<float>("w.attn_sink");
    W.inv = F.up<float>("inv");
    if (comp)
    {
        W.c_wkv = F.up<bf16>("w.compressor.wkv.weight");
        W.c_wgate = F.up<bf16>("w.compressor.wgate.weight");
        W.c_norm = F.up<bf16>("w.compressor.norm.weight");
    }
    const bf16* y = F.up<bf16>("y");
    if (g_gv)
    {
        W.wq_a.gv = make_records(W.wq_a.w, W.wq_a.s, QA, HID);
        W.wq_b.gv = make_records(W.wq_b.w, W.wq_b.s, (long)heads * HD, QA);
        W.wkv.gv = make_records(W.wkv.w, W.wkv.s, HD, HID);
        W.wo_a.gv = make_records(W.wo_a.w, W.wo_a.s, (long)G * OLORA, GROUP_IN);
        W.wo_b.gv = make_records(W.wo_b.w, W.wo_b.s, HID, G * OLORA);
        g_fn_xfrag = gv_xfrag_launch;
        g_fn_gemv = gv_gemv_launch;
        g_use_gv = true;
        printf("--gv: GEMV records built for the five projections; stages at <= 8 rows (decode) run through the GEMV\n");
    }

    // ---- expected tensors (host)
    auto e_qa = F.host<uint16_t>("exp.qa"), e_qr = F.host<uint16_t>("exp.qr"), e_q0 = F.host<uint16_t>("exp.q0"),
         e_q = F.host<uint16_t>("exp.q"), e_kv0 = F.host<uint16_t>("exp.kv0"), e_kv = F.host<uint16_t>("exp.kv"),
         e_o = F.host<uint16_t>("exp.o_pre"), e_orot = F.host<uint16_t>("exp.o_rot"), e_oa = F.host<uint16_t>("exp.oa"),
         e_out = F.host<uint16_t>("exp.out");

    Scratch S;
    void* sbase = dalloc<uint8_t>(Scratch::bytes(n, heads));
    S.carve(sbase, n, heads);
    Cache C{};
    C.ring = dalloc<bf16>((size_t)WIN * HD);
    int R = comp ? (int)F.info("exp.comp").shape[0] : 0;
    if (comp)
    {
        C.comp = dalloc<bf16>((size_t)(R + 1) * HD);
        C.kvs = dalloc<float>(2 * HD);
        C.scs = dalloc<float>(2 * HD);
    }

    printf("---- teacher-forced stages\n");
    // 1 qa: act_quant(y) -> wq_a
    timed("act_quant y", [&] { launch_act_quant(y, HID, S.yq, HID, S.ys, HID / 32, n, HID, st); });
    timed("gemm wq_a", [&] { launch_gemm(S.yq, HID, S.ys, HID / 32, W.wq_a, S.qa, QA, n, 1, 0, 0, 0, st); });
    cmp_bf16("qa = wq_a(act(y))", S.qa, e_qa.data(), (long)n * QA);
    // 2 qr: rmsnorm(expected qa)
    bf16* t_qa = dbf(e_qa);
    bf16* t_qr = dalloc<bf16>((size_t)n * QA);
    timed("rmsnorm q", [&] { k_rmsnorm<<<n, 256, 0, st>>>(t_qa, QA, W.q_norm, t_qr, QA, QA, EPS); });
    cmp_bf16("qr = q_norm(qa)", t_qr, e_qr.data(), (long)n * QA);
    // 3 q0: act_quant(expected qr) -> wq_b
    bf16* t_qr2 = dbf(e_qr);
    bf16* t_q0 = dalloc<bf16>((size_t)n * heads * HD);
    timed("act_quant qr + gemm wq_b", [&] {
        launch_act_quant(t_qr2, QA, S.qrq, QA, S.qrs, QA / 32, n, QA, st);
        launch_gemm(S.qrq, QA, S.qrs, QA / 32, W.wq_b, t_q0, heads * HD, n, 1, 0, 0, 0, st);
    });
    cmp_bf16("q0 = wq_b(act(qr))", t_q0, e_q0.data(), (long)n * heads * HD);
    // 4 q: rope(expected q0)
    bf16* t_q = dalloc<bf16>((size_t)n * heads * HD);
    {
        bf16* d_q0e = dbf(e_q0);
        timed("rope q (+copy)", [&] {
            CK(cudaMemcpyAsync(t_q, d_q0e, (size_t)n * heads * HD * 2, cudaMemcpyDeviceToDevice, st));
            k_rope_tail<<<(unsigned)(((long)n * heads * 32 + 255) / 256), 256, 0, st>>>(t_q, (long)heads * HD, n, heads, 0, nullptr, W.inv, 0);
        });
        cmp_bf16("q = rope(q0)", t_q, e_q.data(), (long)n * heads * HD);
    }
    // 5 kv0: wkv from the act-quantized y
    timed("gemm wkv", [&] { launch_gemm(S.yq, HID, S.ys, HID / 32, W.wkv, S.kv0, HD, n, 1, 0, 0, 0, st); });
    cmp_bf16("kv0 = wkv(act(y))", S.kv0, e_kv0.data(), (long)n * HD);
    // 6 kv rows: norm + rope + fp8 sim, from the expected kv0
    {
        bf16* d_kv0e = dbf(e_kv0);
        bf16* d_kv = dalloc<bf16>((size_t)n * HD);
        timed("kv_prep", [&] { k_kv_prep<<<n, 512, 0, st>>>(d_kv0e, W.kv_norm, W.inv, 0, nullptr, n, d_kv, C.ring, WIN); });
        cmp_bf16("kv rows = fp8sim(rope(norm))", d_kv, e_kv.data(), (long)n * HD);
        std::vector<uint16_t> ring_e((size_t)WIN * HD, 0);
        for (int pos = std::max(0, n - WIN); pos < n; pos++) memcpy(&ring_e[(size_t)(pos % WIN) * HD], &e_kv[(size_t)pos * HD], HD * 2);
        std::vector<uint16_t> ring_g((size_t)WIN * HD);
        CK(cudaMemcpy(ring_g.data(), C.ring, ring_g.size() * 2, cudaMemcpyDeviceToHost));
        report("ring (slot = pos % 128)", stats_bf16(ring_g.data(), ring_e.data(), (long)WIN * HD), 3e-3);
    }

    // compressed branch
    bf16* d_kv_e = dbf(e_kv);   // window rows for the attention stage
    bf16* d_comp_e = nullptr;
    int* d_topk = nullptr;
    if (comp)
    {
        auto e_ckv = F.host<float>("exp.ckv"), e_csc = F.host<float>("exp.csc");
        auto e_crope = F.host<uint16_t>("exp.crope"), e_comp = F.host<uint16_t>("exp.comp");
        timed("f32gemm2 (wkv, wgate)", [&] {
            k_f32gemm2<<<dim3(HD / FB_N, (n + FB_M - 1) / FB_M, 2), 256, 0, st>>>(y, W.c_wkv, W.c_wgate, S.ckv, S.csc, n, HD, HID);
        });
        // the twin's ckv / csc hold whole pairs only (n odd: the last token is the pending one), the kernels' [n] rows one more
        const long pairs_rows = (long)F.info("exp.ckv").shape[0];
        cmp_f32("ckv = c_wkv(y) f32", S.ckv, e_ckv.data(), pairs_rows * HD);
        cmp_f32("csc = c_wgate(y) f32", S.csc, e_csc.data(), pairs_rows * HD);
        e_ckv.resize((size_t)n * HD, 0.0f);
        e_csc.resize((size_t)n * HD, 0.0f);
        float* d_ckv = dcopy(e_ckv);
        float* d_csc = dcopy(e_csc);
        bf16* d_lat = dalloc<bf16>((size_t)R * HD);
        bf16* d_ck = dalloc<bf16>((size_t)(R + 1) * HD);
        bf16* d_cn = dalloc<bf16>((size_t)R * HD);
        auto e_cnorm = F.host<uint16_t>("exp.cnorm");
        timed("comp_pool", [&] { k_comp_pool<<<(n + 1) / 2, 512, 0, st>>>(d_ckv, d_csc, n, 0, W.c_norm, W.inv, d_ck, C.kvs, C.scs, d_lat, d_cn); });
        cmp_bf16("comp post-norm pre-rope", d_cn, e_cnorm.data(), (long)R * HD);
        cmp_bf16("comp pre-fp4 (rope'd)", d_lat, e_crope.data(), (long)R * HD);
        cmp_bf16("comp rows (fp4 sim)", d_ck, e_comp.data(), (long)R * HD);
        d_comp_e = dbf(e_comp);
        // topk from the fixture's id lists: [window slots | compressed ids offset by n]
        auto ids = F.host<int32_t>("exp.ids");
        const int cols = WL + TK;
        std::vector<int32_t> tk((size_t)n * TK);
        long wbad = 0;
        for (int t = 0; t < n; t++)
        {
            for (int i = 0; i < WL; i++)   // official window list: max(t-127, 0) + i, -1 when beyond t
            {
                const int want = std::max(t - (WIN - 1), 0) + i;
                const int got = ids[(size_t)t * cols + i];
                if (want <= t ? got != want : got != -1) wbad++;
            }
            for (int j = 0; j < TK; j++)
            {
                const int v = ids[(size_t)t * cols + WL + j];
                tk[(size_t)t * TK + j] = v >= 0 ? v - n : -1;
            }
        }
        printf("%-30s %s (%ld mismatches)\n", "window id order vs fixture", wbad ? "BAD" : "EXACT", wbad);
        g_bad += wbad != 0;
        d_topk = dcopy(tk);
    }

    // 9 attention core, fed by expected q / kv / compressed rows
    {
        bf16* d_qe = dbf(e_q);
        bf16* d_o = dalloc<bf16>((size_t)n * heads * HD);
        AttnP a{};
        a.q = d_qe; a.kvw = d_kv_e; a.ck = d_comp_e; a.topk = d_topk; a.sink = W.sink; a.out = d_o;
        a.ldtk = TK; a.n = n; a.heads = heads; a.start = 0; a.decode = 0; a.WL = WL; a.TK = TK; a.scale = SOFTMAX_SCALE;
        timed("attention", [&] { launch_attn(a, st); });
        cmp_bf16("o_pre = attn(q, kv, comp, sink)", d_o, e_o.data(), (long)n * heads * HD);
    }
    // 10 o prep from the expected o
    bf16* d_oe = dbf(e_o);
    {
        bf16* d_orot = dalloc<bf16>((size_t)n * heads * HD);
        timed("o_prep", [&] { k_o_prep<<<n * heads, 512, 0, st>>>(d_oe, heads, 0, nullptr, W.inv, d_orot, S.oq, S.os); });
        cmp_bf16("o_rot = inv_rope(o)", d_orot, e_orot.data(), (long)n * heads * HD);
    }
    // 11 o_a grouped gemm from that quantized o
    timed("gemm wo_a (grouped)", [&] {
        launch_gemm(S.oq, heads * HD, S.os, heads * HD / 32, W.wo_a, S.oa, G * OLORA, n, G, GROUP_IN, GROUP_IN / 32, OLORA, st);
    });
    cmp_bf16("oa = wo_a(act(o_rot))", S.oa, e_oa.data(), (long)n * G * OLORA);
    // 12 o_b from the expected oa
    {
        bf16* d_oae = dbf(e_oa);
        bf16* d_out = dalloc<bf16>((size_t)n * HID);
        timed("act_quant oa + gemm wo_b", [&] {
            launch_act_quant(d_oae, G * OLORA, S.oaq, G * OLORA, S.oas, G * OLORA / 32, n, G * OLORA, st);
            launch_gemm(S.oaq, G * OLORA, S.oas, G * OLORA / 32, W.wo_b, d_out, HID, n, 1, 0, 0, 0, st);
        });
        cmp_bf16("out = wo_b(act(oa))", d_out, e_out.data(), (long)n * HID);
    }

    // ---- free-running layer
    printf("---- layer_forward (free-running)\n");
    {
        bf16* d_out = dalloc<bf16>((size_t)n * HID);
        CK(cudaMemset(C.ring, 0, (size_t)WIN * HD * 2));
        if (comp) CK(cudaMemset(C.comp, 0, (size_t)(R + 1) * HD * 2));
        timed("layer_forward", [&] { layer_forward(W, C, comp ? C.comp : nullptr, S, y, n, 0, heads, d_topk, TK, TK, d_out, st); });
        // Free-running, a last-ulp gemm difference upstream is amplified at each fp8 re-quantization (a rounding flip moves one
        // value by a whole e4m3 step), so the downstream tolerances are wider than the teacher-forced ones.
        cmp_bf16("e2e qa", S.qa, e_qa.data(), (long)n * QA);
        cmp_bf16("e2e q", S.q, e_q.data(), (long)n * heads * HD);
        cmp_bf16("e2e kv rows", S.kvrows, e_kv.data(), (long)n * HD, 2e-3);
        if (comp)
        {
            auto e_comp = F.host<uint16_t>("exp.comp");
            cmp_bf16("e2e comp rows", C.comp, e_comp.data(), (long)R * HD, 2e-2);
            auto e_cnorm = F.host<uint16_t>("exp.cnorm");
            cmp_bf16("e2e comp post-norm", S.cnorm, e_cnorm.data(), (long)R * HD, 2e-2);
        }
        cmp_bf16("e2e o_pre", S.o, e_o.data(), (long)n * heads * HD, 2e-3);
        cmp_bf16("e2e oa", S.oa, e_oa.data(), (long)n * G * OLORA, 6e-3);
        cmp_bf16("e2e OUT", d_out, e_out.data(), (long)n * HID, 1.5e-2);
        dump_bf16(layer, "qr", S.qr, (long)n * QA);
        dump_bf16(layer, "q", S.q, (long)n * heads * HD);
        dump_bf16(layer, "kv", S.kvrows, (long)n * HD);
        dump_bf16(layer, "o_sparse", S.o, (long)n * heads * HD);
        dump_bf16(layer, "wo_a_out", S.oa, (long)n * G * OLORA);
        dump_bf16(layer, "attn_out", d_out, (long)n * HID);
        if (comp) dump_bf16(layer, "lat", C.comp, (long)(n / 2) * HD);
    }

    // ---- decode steps (fixture from `attn_ref_run.py decode`): prefill n rows, D one-row steps at positions n .. n+D-1
    if (F.D > 0 && F.has("d.y"))
    {
        const int D = F.D;
        const int cols = (int)F.info("d.ids").shape[1];
        const int TKd = cols - WIN;
        const int Rall = comp ? (int)F.info("d.comp").shape[0] : 0;
        printf("---- decode: prefill %d rows, then %d steps at positions %d..%d (TKd=%d, %d compressed rows at the end)\n", n, D, n, n + D - 1, TKd, Rall);
        const bf16* d_dy = F.up<bf16>("d.y");
        auto e_dq = F.host<uint16_t>("d.q"), e_dkv = F.host<uint16_t>("d.kv"), e_do = F.host<uint16_t>("d.o"), e_dout = F.host<uint16_t>("d.out");
        auto dids = F.host<int32_t>("d.ids");
        // topk of every step: twin ids are offset by the 128 ring slots; -1 stays -1. Window part must equal slot(pos) lists.
        std::vector<int32_t> tkd((size_t)D * std::max(TKd, 1), -1);
        long wbad = 0;
        for (int k = 0; k < D; k++)
        {
            const int pos = n + k;
            for (int i = 0; i < WIN; i++)
            {
                const int p = pos - (WIN - 1) + i;
                const int want = p >= 0 ? p % WIN : -1;
                if (dids[(size_t)k * cols + i] != want) wbad++;
            }
            for (int j = 0; j < TKd; j++)
            {
                const int v = dids[(size_t)k * cols + WIN + j];
                tkd[(size_t)k * TKd + j] = v >= 0 ? v - WIN : -1;
            }
        }
        printf("%-30s %s (%ld mismatches)\n", "decode window slot lists", wbad ? "BAD" : "EXACT", wbad);
        g_bad += wbad != 0;
        const int* d_tkd = comp ? dcopy(tkd) : nullptr;
        std::vector<uint16_t> e_dcomp, e_dcn;
        if (comp)
        {
            e_dcomp = F.host<uint16_t>("d.comp");
            if (F.has("d.cnorm")) e_dcn = F.host<uint16_t>("d.cnorm");
        }
        auto get = [&](const void* dptr, long cnt) { std::vector<uint16_t> g(cnt); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(g.data(), dptr, cnt * 2, cudaMemcpyDeviceToHost)); return g; };
        auto fresh = [&](int RR, bf16* d_pre_out) {   // new caches of RR ring rows, prefill the n rows
            Cache C2{};
            C2.ring = dalloc<bf16>((size_t)RR * HD);
            C2.ring_rows = RR;
            if (comp) { C2.comp = dalloc<bf16>((size_t)(Rall + 2) * HD); C2.kvs = dalloc<float>(2 * HD); C2.scs = dalloc<float>(2 * HD); }
            layer_forward(W, C2, comp ? C2.comp : nullptr, S, y, n, 0, heads, d_topk, TK, TK, d_pre_out, st);
            CK(cudaDeviceSynchronize());
            return C2;
        };
        std::vector<uint16_t> seq_out_ref;   // the R = 128 sequential outputs, compared bitwise with the other runs
        for (int RR : {WIN, 136})
        {
            bf16* d_pre = dalloc<bf16>((size_t)n * HID);
            Cache C2 = fresh(RR, d_pre);
            bf16* d_so = dalloc<bf16>((size_t)D * HID);
            std::vector<uint16_t> kq, kv, ko;
            size_t cnrow = 0;
            printf("sequential steps, ring_rows=%d\n", RR);
            for (int k = 0; k < D; k++)
            {
                const int pos = n + k;
                layer_forward(W, C2, comp ? C2.comp : nullptr, S, d_dy + (size_t)k * HID, 1, pos, heads, comp ? d_tkd + (size_t)k * TKd : nullptr, TKd, TKd,
                              d_so + (size_t)k * HID, st);
                Stats sq = stats_bf16(get(S.q, (long)heads * HD).data(), e_dq.data() + (size_t)k * heads * HD, (long)heads * HD);
                Stats skv = stats_bf16(get(S.kvrows, HD).data(), e_dkv.data() + (size_t)k * HD, HD);
                Stats so = stats_bf16(get(S.o, (long)heads * HD).data(), e_do.data() + (size_t)k * heads * HD, (long)heads * HD);
                Stats sout = stats_bf16(get(d_so + (size_t)k * HID, HID).data(), e_dout.data() + (size_t)k * HID, HID);
                const bool ok = !sq.nonfinite && !skv.nonfinite && !so.nonfinite && !sout.nonfinite && sq.rrms < 3e-3 && skv.rrms < 2e-3 && so.rrms < 4e-3 && sout.rrms < 1.5e-2;
                g_bad += !ok;
                printf("  step %d pos %-4d %-4s q rrms=%.2e kv eq=%.4f o rrms=%.2e out rrms=%.2e (eq=%.4f ulp1=%.4f)\n", k, pos, ok ? "ok" : "BAD", sq.rrms, skv.eq, so.rrms,
                       sout.rrms, sout.eq, sout.ulp1);
                if (comp && (pos & 1) && cnrow < e_dcn.size() / HD)
                {
                    Stats sc = stats_bf16(get(S.cnorm, HD).data(), e_dcn.data() + cnrow * HD, HD);
                    cnrow++;
                    if (sc.rrms > 2e-2 || sc.nonfinite) { g_bad++; printf("    comp post-norm row BAD rrms=%.2e\n", sc.rrms); }
                }
            }
            if (comp) cmp_bf16("comp rows after decode", C2.comp, e_dcomp.data(), (long)Rall * HD, 2e-2);
            auto so_h = get(d_so, (long)D * HID);
            if (RR == WIN) seq_out_ref = so_h;
            else
            {
                Stats s = stats_bf16(so_h.data(), seq_out_ref.data(), (long)D * HID);
                report("ring 136 vs ring 128 (bitwise)", s, 0.0);
                g_bad += s.eq != 1.0;
            }
            if (RR == 136)
            {   // verify chunk: the D rows as one call at start > 0 on a fresh prefill; must equal the sequential steps
                bf16* d_pre2 = dalloc<bf16>((size_t)n * HID);
                Cache C3 = fresh(136, d_pre2);
                bf16* d_co = dalloc<bf16>((size_t)D * HID);
                layer_forward(W, C3, comp ? C3.comp : nullptr, S, d_dy, D, n, heads, d_tkd, TKd, TKd, d_co, st);
                auto co_h = get(d_co, (long)D * HID);
                Stats sb = stats_bf16(co_h.data(), seq_out_ref.data(), (long)D * HID);
                report("verify chunk vs sequential (bitwise)", sb, 0.0);
                g_bad += sb.eq != 1.0;
                report("verify chunk vs twin out", stats_bf16(co_h.data(), e_dout.data(), (long)D * HID), 1.5e-2);
                Stats sq = stats_bf16(get(S.q, (long)D * heads * HD).data(), e_dq.data(), (long)D * heads * HD);
                report("verify chunk q vs twin", sq, 3e-3);
                if (comp) cmp_bf16("verify chunk comp rows", C3.comp, e_dcomp.data(), (long)Rall * HD, 2e-2);
                {   // the engine form: start read from a device word, TK = 512 with a -1 tail; must equal the host-start runs BITWISE
                    int* d_pos = dalloc<int>(1);
                    std::vector<int> tk512((size_t)D * 512, -1);
                    if (comp) for (int k = 0; k < D; k++) for (int j = 0; j < TKd; j++) tk512[(size_t)k * 512 + j] = tkd[(size_t)k * TKd + j];
                    const int* d_tk512 = dcopy(tk512);
                    bf16* d_pre3 = dalloc<bf16>((size_t)n * HID);
                    Cache C4 = fresh(136, d_pre3);
                    bf16* d_dso = dalloc<bf16>((size_t)D * HID);
                    for (int k = 0; k < D; k++)
                    {
                        const int pos = n + k;
                        CK(cudaMemcpy(d_pos, &pos, 4, cudaMemcpyHostToDevice));
                        layer_forward(W, C4, comp ? C4.comp : nullptr, S, d_dy + (size_t)k * HID, 1, 0, heads, comp ? d_tk512 + (size_t)k * 512 : nullptr, 512,
                                      comp ? 512 : 0, d_dso + (size_t)k * HID, st, d_pos);
                    }
                    Stats sd = stats_bf16(get(d_dso, (long)D * HID).data(), seq_out_ref.data(), (long)D * HID);
                    report("device-start steps vs host-start (bitwise)", sd, 0.0);
                    g_bad += sd.eq != 1.0;
                    if (comp) cmp_bf16("device-start comp rows", C4.comp, e_dcomp.data(), (long)Rall * HD, 2e-2);
                    bf16* d_pre4 = dalloc<bf16>((size_t)n * HID);
                    Cache C5 = fresh(136, d_pre4);
                    bf16* d_co2 = dalloc<bf16>((size_t)D * HID);
                    CK(cudaMemcpy(d_pos, &n, 4, cudaMemcpyHostToDevice));
                    layer_forward(W, C5, comp ? C5.comp : nullptr, S, d_dy, D, 0, heads, d_tk512, 512, comp ? 512 : 0, d_co2, st, d_pos);
                    Stats sc2 = stats_bf16(get(d_co2, (long)D * HID).data(), seq_out_ref.data(), (long)D * HID);
                    report("device-start chunk vs sequential (bitwise)", sc2, 0.0);
                    g_bad += sc2.eq != 1.0;
                    if (comp) cmp_bf16("device-start chunk comp rows", C5.comp, e_dcomp.data(), (long)Rall * HD, 2e-2);
                }
            }
        }
    }
    if (g_gv) gv_section(W, G);
    printf("%s (%d stage(s) BAD)\n", g_bad ? "FAIL" : "PASS", g_bad);
    return g_bad ? 1 : 0;
}
}   // namespace ds41a

int main(int argc, char** argv) { return ds41a::test_main(argc, argv); }
