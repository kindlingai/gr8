// The Inkling 2 Small speed paths on the device, each against the stock path it replaces.
//
//   1. The head's candidates packed on the device (kernels/ink2_head.cuh) against ink2rt::head_pack on the host:
//      rows of logits with -inf cells, runs of equal logits, both zeros, fewer finite logits than candidates and
//      none at all; 16, 256 and 512 candidates. The maximum, the count and every candidate: exact. The sum: to 1e-12.
//      Also BlasScope / ink2rt::blas_gemm directly against stock k_ink2_gemm on BF16 and fp32 projections (cosine
//      >= 0.999999, tight cellwise and relative bounds), and bit-exact fallback to k_ink2_gemm for small/odd N
//      (such as TP=4 sink N = 9) and unaligned pointers.
//   2. The tensor-core attention of a prefill chunk (attn/ink2_attn_tc.cuh) against k_ink2_attn_core at the full
//      model's head geometry (head_dim 128, rel_dim 16): a local layer whose keys come from the ring and from the
//      chunk, a rank whose first K / V head has part of its query heads, and a global layer on pages. 40 rows (a
//      part group of 16). A cell may differ by max(|x|, 1) / 32 and cosine must be at least 0.9999 (the sums are in
//      another order and BF16 rounds the output); the largest difference and cosine are printed.
//   3. The engine on a miniature checkpoint (ink2_mini_ckpt.h), 1, 4 and 6 simulated ranks, a 70-token prefill in
//      chunks of 48 and 22 with every row sampled, then 8 single rows (plus a 3-draft MTP speculative round on each
//      rank count): the device head pack on and off give the same tokens, drafted tokens and candidates, exact; a
//      24-row chunk (M < 32) with all speed paths on is also bit-exact against stock; with cuBLAS on as well, the
//      device head pack (nc = 64) matches the full-vocab cuBLAS run (nc = 768) bit for bit on top-64 candidates,
//      tokens and MTP drafts, and all 750 valid vocabulary logits per row satisfy the 6-layer NVFP4/MXFP8 MoE trunk
//      bounds (cos >= 0.85, rel <= 0.55, per-row cos >= 0.75) separately on the 48-row cuBLAS chunk, the 22-row tail
//      chunk, and the 8 decode steps. (The miniature's head_dim is 16, so the tensor-core kernel is not on this path;
//      check 2 covers it.)
//   4. The device all-reduce (kernels/ink2_fabric.cuh) as rank 1 of a 4-rank fabric in pinned host memory, the test
//      playing the peers (before the launches, or after the doorbell while the device spins): 1, 8, 64 and 512 rows
//      of 2304 and 4096 cells; the sum's bits are the stock sum's. Also the slot offsets and the credits of the pooled
//      96-collective layout. The proxy and the wire of a real multi-rank run are not on this path.
//
// --bench: also times the head pack (8 rows of 50304 logits, 256 candidates), cuBLAS vs k_ink2_gemm (128 and 512 rows
// of 4608 -> 6144), and the two attention kernels (512 rows at ctx 2048 local and ctx 8192 global) and prints ms.
//
// Build (from engine/runtime, after scripts/build.sh has built the Rust libraries and bin/obj/g6pack.o):
//   nvcc -O3 -std=c++17 -Xcompiler -ffp-contract=off -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_INK2=1 -DDEC_S2 \
//     -DG6_RS_LOADER -I . -I ../kernels -I ../attn -I ../ref/ink2 -o ../../bin/ink2_speed_test ../tests/ink2_speed_test.cu \
//     ../../bin/obj/g6pack.o ../loader-rs/target/release/libg6load.a ../knobs-rs/target/release/libgr8knobs.a \
//     ../target/release/libg6host.a -lcublas -lpthread -ldl -lm
// Run: bin/ink2_speed_test [--bench]   (one GPU; prints PASS and returns 0)
#include "../comms/fabric.h"
#include "../kernels/ink2_fabric.cuh"
#include "../runtime/ink2_step.cuh"
#include "ink2_mini_ckpt.h"
#include <chrono>
#include <filesystem>
#include <memory>

using namespace ink2mini;
using ink2rt::Engine;
using ink2rt::Options;
using ink2rt::RowOut;
using ink2rt::RowSample;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) { fprintf(stderr, "FATAL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } \
    } while (0)

static int fails = 0;
static void check(bool ok, const std::string& what)
{
    if (ok) return;
    ++fails;
    if (fails <= 40) fprintf(stderr, "FAIL %s\n", what.c_str());
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint16_t f2bf(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
static float bf2f(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float unit(Rng& r) { return static_cast<float>(r.next() & 0xffffff) / 8388608.0f - 1.0f; }   // [-1, 1)
template <class T> static T* to_dev(const std::vector<T>& v)
{
    T* d = nullptr;
    CK(cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)));
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static std::vector<uint16_t> rand_bf(Rng& r, size_t n, float scale)
{
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = f2bf(unit(r) * scale);
    return v;
}

// ---------------------------------------------------------------------------------------------- 1. the head pack
static void check_head_pack(int vr, int NC, int rows, bool bench)
{
    Rng r{0x68656164u + static_cast<uint64_t>(vr) * 31 + static_cast<uint64_t>(NC)};
    std::vector<float> lg(static_cast<size_t>(rows) * vr);
    for (int m = 0; m < rows; ++m)
        for (int j = 0; j < vr; ++j)
        {
            float v = unit(r) * 12.0f;
            const uint32_t k = r.next();
            if (m % 4 == 1) v = static_cast<float>(static_cast<int>(v));           // many equal logits
            if (m % 4 == 2 && k % 3) v = -INFINITY;                                // a third finite
            if (m % 4 == 3 && j >= 5) v = -INFINITY;                               // fewer finite than candidates
            if (m == rows - 1) v = -INFINITY;                                      // none
            if (m % 4 == 1 && k % 97 == 0) v = k & 256 ? 0.0f : -0.0f;             // both zeros
            if (m == 0 && j % 1000 == 7) v = 11.5f;                                // a run of equal maxima across the threads' chunks
            lg[static_cast<size_t>(m) * vr + j] = v;
        }
    const size_t rb = ink2rt::head_row_bytes(NC);
    const int v0 = 3 * vr;
    std::vector<uint8_t> ref(static_cast<size_t>(rows) * rb), got(ref.size());
    double t0 = now_ms();
    for (int m = 0; m < rows; ++m) ink2rt::head_pack(&lg[static_cast<size_t>(m) * vr], vr, v0, NC, &ref[static_cast<size_t>(m) * rb]);
    const double t_cpu = now_ms() - t0;
    float* d_lg = to_dev(lg);
    uint8_t* d_out = to_dev(got);
    ink2::head_pack_dev(d_lg, rows, vr, v0, NC, d_out, rb, 0);
    CK(cudaDeviceSynchronize());
    t0 = now_ms();
    ink2::head_pack_dev(d_lg, rows, vr, v0, NC, d_out, rb, 0);
    CK(cudaMemcpy(got.data(), d_out, got.size(), cudaMemcpyDeviceToHost));
    const double t_gpu = now_ms() - t0;
    for (int m = 0; m < rows; ++m)
    {
        const uint8_t *a = &ref[static_cast<size_t>(m) * rb], *b = &got[static_cast<size_t>(m) * rb];
        const std::string w = "head pack vr " + std::to_string(vr) + " nc " + std::to_string(NC) + " row " + std::to_string(m);
        double sa, sb;
        memcpy(&sa, a + 8, 8);
        memcpy(&sb, b + 8, 8);
        check(memcmp(a, b, 8) == 0, w + ": maximum or count");
        check(std::fabs(sa - sb) <= 1e-12 * std::max(1.0, std::fabs(sa)), w + ": sum " + std::to_string(sa) + " / " + std::to_string(sb));
        check(memcmp(a + 16, b + 16, rb - 16) == 0, w + ": candidates");
    }
    if (bench) printf("   bench head pack: %d rows of %d logits, %d candidates: host %.3f ms (without its copy of the logits), device %.3f ms\n", rows, vr, NC, t_cpu, t_gpu);
    CK(cudaFree(d_lg));
    CK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------------------------- 2. the attention
struct AttnCase
{
    const char* name;
    int M, p0, nh_q, q0, kv0, nh_kv, window, global;
};
static void check_attn_tc(const AttnCase& tc, bool bench)
{
    const int hd = ink2::TC_HD, rd = ink2::TC_RD, gqa = 6, rel_win = 1024, base = 6, M = tc.M, nk = tc.nh_kv * hd;
    Rng r{0x7463u + static_cast<uint64_t>(tc.p0) * 131 + static_cast<uint64_t>(tc.nh_q)};
    const int pmax = tc.p0 + M - 1, max_pps = pmax / ink2::KV_PAGE_TOKENS + 1;
    std::vector<int> slot(static_cast<size_t>(M), 0), pos(static_cast<size_t>(M)), run0(static_cast<size_t>(M), 0), pt(static_cast<size_t>(max_pps));
    for (int m = 0; m < M; ++m) pos[m] = tc.p0 + m;
    for (int i = 0; i < max_pps; ++i) pt[i] = max_pps - 1 - i;   // the pages are not in position order
    const std::vector<uint16_t> q_raw = rand_bf(r, static_cast<size_t>(M) * tc.nh_q * hd, 1.5f), r_raw = rand_bf(r, static_cast<size_t>(M) * tc.nh_q * rd, 1.5f),
                                gate = rand_bf(r, static_cast<size_t>(M) * tc.nh_q * hd, 2.0f), sink = rand_bf(r, static_cast<size_t>(M) * tc.nh_q, 2.0f),
                                q_clip = rand_bf(r, static_cast<size_t>(tc.nh_q) * hd, 1.2f), rel_clip = rand_bf(r, static_cast<size_t>(tc.nh_q) * rd, 1.2f),
                                rel_proj = rand_bf(r, static_cast<size_t>(rd) * rel_win, 0.6f), kv_new = rand_bf(r, static_cast<size_t>(M) * 2 * nk, 0.5f),
                                ring = rand_bf(r, static_cast<size_t>(2) * tc.window * nk, 0.5f),
                                pages = rand_bf(r, static_cast<size_t>(max_pps) * 2 * ink2::KV_PAGE_TOKENS * nk, 0.5f);
    const int smax = ink2::attn_splits(tc.global != 0, tc.window, pmax);
    const size_t ncell = static_cast<size_t>(M) * tc.nh_q * hd;
    uint16_t *d_q = to_dev(q_raw), *d_r = to_dev(r_raw), *d_gate = to_dev(gate), *d_sink = to_dev(sink), *d_qc = to_dev(q_clip), *d_rc = to_dev(rel_clip), *d_rp = to_dev(rel_proj),
             *d_kv = to_dev(kv_new), *d_ring = to_dev(ring), *d_pages = to_dev(pages);
    int *d_slot = to_dev(slot), *d_pos = to_dev(pos), *d_run0 = to_dev(run0), *d_pt = to_dev(pt);
    uint16_t* d_og = to_dev(std::vector<uint16_t>(ncell, 0));
    float* d_part = to_dev(std::vector<float>(static_cast<size_t>(M) * tc.nh_q * smax * (hd + 2), 0.0f));
    ink2::AttnCoreArgs a{d_q, d_r, d_gate, d_sink, d_qc, d_rc, d_rp, d_kv, d_ring, d_pages, d_pt, ink2::Rows{d_slot, d_pos, d_run0}, d_og, d_part,
                         0, M, tc.nh_q, tc.q0, gqa, tc.kv0, nk, hd, rd, rel_win, tc.window, base, tc.global, max_pps, smax};
    std::vector<uint16_t> ref(ncell), got(ncell);
    CK(cudaDeviceSynchronize());
    double t0 = now_ms();
    ink2::k_ink2_attn_core<<<dim3(M * tc.nh_q, smax), ink2::ACT>>>(a);
    if (smax > 1) ink2::k_ink2_attn_merge<<<M * tc.nh_q, ink2::ACT>>>(a);
    CK(cudaDeviceSynchronize());
    const double t_ref = now_ms() - t0;
    CK(cudaMemcpy(ref.data(), d_og, ncell * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemset(d_og, 0, ncell * 2));
    CK(cudaFuncSetAttribute(ink2::k_ink2_attn_prefill_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(ink2::attn_tc_smem(ink2::TC_MAX_GQ))));
    t0 = now_ms();
    ink2::k_ink2_attn_qr_prep<<<M * tc.nh_q, ink2::ACT>>>(d_q, d_r, d_qc, d_rc, a.rw, tc.nh_q, hd, rd, base, tc.global);
    ink2::k_ink2_attn_prefill_tc<<<dim3((M + 15) / 16, tc.nh_kv), gqa * 32, ink2::attn_tc_smem(gqa)>>>(a, gqa);
    CK(cudaDeviceSynchronize());
    const double t_tc = now_ms() - t0;
    CK(cudaMemcpy(got.data(), d_og, ncell * 2, cudaMemcpyDeviceToHost));
    double worst = 0, dot = 0, na = 0, nb = 0;
    size_t bad = 0, differ = 0;
    for (size_t i = 0; i < ncell; ++i)
    {
        const double x = bf2f(ref[i]), y = bf2f(got[i]);
        const double d = std::fabs(x - y), tol = std::max(1.0, std::fabs(x)) / 32;
        if (!(d <= tol)) ++bad;   // a NaN is a failure
        if (ref[i] != got[i]) ++differ;
        if (d > worst) worst = d;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    const double cos = na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0;
    printf("   %s: %zu cells, %zu differ, largest difference %.5f, cosine %.6f%s\n", tc.name, ncell, differ, worst, cos, (bad || !(cos >= 0.9999)) ? "  <-- FAIL" : "");
    check(bad == 0, std::string("tensor-core attention, ") + tc.name + ": " + std::to_string(bad) + " cells past the tolerance");
    check(cos >= 0.9999, std::string("tensor-core attention, ") + tc.name + ": cosine " + std::to_string(cos));
    if (bench) printf("   bench attention %s: %d rows, stock %.3f ms, tensor cores %.3f ms\n", tc.name, M, t_ref, t_tc);
    for (void* p : {static_cast<void*>(d_q), static_cast<void*>(d_r), static_cast<void*>(d_gate), static_cast<void*>(d_sink), static_cast<void*>(d_qc), static_cast<void*>(d_rc),
                    static_cast<void*>(d_rp), static_cast<void*>(d_kv), static_cast<void*>(d_ring), static_cast<void*>(d_pages), static_cast<void*>(d_slot), static_cast<void*>(d_pos),
                    static_cast<void*>(d_run0), static_cast<void*>(d_pt), static_cast<void*>(d_og), static_cast<void*>(d_part)})
        CK(cudaFree(p));
}

// ---------------------------------------------------------------------------------------------- 3. the engine
constexpr int CAP = 192, ROWS = 48, NCAND = 64, PROMPT = 70, SINGLES = 8;
constexpr double E2E_SEQ_COS = 0.85, E2E_SEQ_REL = 0.55, E2E_ROW_COS = 0.75;

static void check_blas_gemm()
{
    cublasHandle_t hd = nullptr;
    if (cublasCreate(&hd) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "FATAL cublasCreate\n"); exit(2); }
    Rng r{0x626c6173u};

    // 1b-i. Isolated BF16 projections (M = 48 and unaligned M = 37) through BlasScope vs stock k_ink2_gemm:
    struct Shape { int M, K, N; };
    for (const Shape& s : {Shape{48, 256, 192}, Shape{37, 256, 144}, Shape{64, 192, 256}})
    {
        const size_t n = static_cast<size_t>(s.M) * s.N;
        const std::vector<uint16_t> x = rand_bf(r, static_cast<size_t>(s.M) * s.K, 1.0f), w = rand_bf(r, static_cast<size_t>(s.N) * s.K, 0.25f);
        uint16_t *dx = to_dev(x), *dw = to_dev(w), *d_ref = to_dev(std::vector<uint16_t>(n, 0)), *d_got = to_dev(std::vector<uint16_t>(n, 0));
        ink2::gemm_bf16(dx, dw, s.M, s.K, s.N, d_ref, 0);
        {
            ink2rt::BlasScope bs(hd);
            check(ink2rt::blas_gemm(dx, dw, s.M, s.K, s.N, d_got, ink2::GEMM_BF16, 0),
                  "blas_gemm BF16 " + std::to_string(s.M) + "x" + std::to_string(s.K) + "x" + std::to_string(s.N) + ": did not take cuBLAS");
            CK(cudaMemset(d_got, 0, n * 2));
            ink2::gemm_bf16(dx, dw, s.M, s.K, s.N, d_got, 0);
        }
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> ref(n), got(n);
        CK(cudaMemcpy(ref.data(), d_ref, n * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got.data(), d_got, n * 2, cudaMemcpyDeviceToHost));
        double dot = 0, na = 0, nb = 0, nd = 0, worst = 0, max_abs = 0;
        bool fin = true;
        for (size_t i = 0; i < n; ++i)
        {
            const double a = bf2f(ref[i]), b = bf2f(got[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) { fin = false; continue; }
            dot += a * b; na += a * a; nb += b * b; nd += (a - b) * (a - b);
            worst = std::max(worst, std::fabs(a - b));
            max_abs = std::max(max_abs, std::fabs(a));
        }
        const double cos = na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0;
        const double rel = nb > 0 ? std::sqrt(nd / nb) : 1.0;
        const std::string tag = "blas_gemm BF16 " + std::to_string(s.M) + "x" + std::to_string(s.K) + "x" + std::to_string(s.N);
        check(fin, tag + ": non-finite");
        check(worst <= std::max(1.0, max_abs) / 128.0, tag + ": worst diff " + std::to_string(worst));
        check(rel <= 2e-3, tag + ": rel " + std::to_string(rel));
        check(cos >= 0.999999, tag + ": cos " + std::to_string(cos));
        CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(d_ref)); CK(cudaFree(d_got));
    }

    // 1b-ii. Isolated FP32 partial-sum projections (T > 1 wo path) through BlasScope vs stock k_ink2_gemm:
    for (const Shape& s : {Shape{48, 192, 256}, Shape{37, 144, 256}})
    {
        const size_t n = static_cast<size_t>(s.M) * s.N;
        const std::vector<uint16_t> x = rand_bf(r, static_cast<size_t>(s.M) * s.K, 1.0f), w = rand_bf(r, static_cast<size_t>(s.N) * s.K, 0.25f);
        uint16_t *dx = to_dev(x), *dw = to_dev(w);
        float *d_ref = to_dev(std::vector<float>(n, 0.0f)), *d_got = to_dev(std::vector<float>(n, 0.0f));
        ink2::gemm_f32(dx, dw, s.M, s.K, s.N, d_ref, 0);
        {
            ink2rt::BlasScope bs(hd);
            check(ink2rt::blas_gemm(dx, dw, s.M, s.K, s.N, d_got, ink2::GEMM_F32, 0),
                  "blas_gemm F32 " + std::to_string(s.M) + "x" + std::to_string(s.K) + "x" + std::to_string(s.N) + ": did not take cuBLAS");
            CK(cudaMemset(d_got, 0, n * 4));
            ink2::gemm_f32(dx, dw, s.M, s.K, s.N, d_got, 0);
        }
        CK(cudaDeviceSynchronize());
        std::vector<float> ref(n), got(n);
        CK(cudaMemcpy(ref.data(), d_ref, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got.data(), d_got, n * 4, cudaMemcpyDeviceToHost));
        double dot = 0, na = 0, nb = 0, nd = 0;
        bool fin = true;
        for (size_t i = 0; i < n; ++i)
        {
            const double a = ref[i], b = got[i];
            if (!std::isfinite(a) || !std::isfinite(b)) { fin = false; continue; }
            dot += a * b; na += a * a; nb += b * b; nd += (a - b) * (a - b);
        }
        const double cos = na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0;
        const double rel = nb > 0 ? std::sqrt(nd / nb) : 1.0;
        const std::string tag = "blas_gemm F32 " + std::to_string(s.M) + "x" + std::to_string(s.K) + "x" + std::to_string(s.N);
        check(fin, tag + ": non-finite");
        check(rel <= 1e-4, tag + ": rel " + std::to_string(rel));
        check(cos >= 0.9999999, tag + ": cos " + std::to_string(cos));
        CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(d_ref)); CK(cudaFree(d_got));
    }

    // 1b-iii. Small / odd N (TP=4 sink N = 9, TP=1 sink N = 36, TP=4 wproj N = 8) and unaligned pointer fallback:
    for (int badN : {9, 36})
    {
        const size_t n = static_cast<size_t>(48) * badN;
        const std::vector<uint16_t> x = rand_bf(r, 48 * 256, 1.0f), w = rand_bf(r, static_cast<size_t>(badN) * 256, 0.25f);
        uint16_t *dx = to_dev(x), *dw = to_dev(w), *d_ref = to_dev(std::vector<uint16_t>(n, 0)), *d_got = to_dev(std::vector<uint16_t>(n, 0));
        ink2::gemm_bf16(dx, dw, 48, 256, badN, d_ref, 0);
        {
            ink2rt::BlasScope bs(hd);
            check(!ink2rt::blas_gemm(dx, dw, 48, 256, badN, d_got, ink2::GEMM_BF16, 0), "blas_gemm N=" + std::to_string(badN) + " did not fall back");
            ink2::gemm_bf16(dx, dw, 48, 256, badN, d_got, 0);
        }
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> ref(n), got(n);
        CK(cudaMemcpy(ref.data(), d_ref, n * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got.data(), d_got, n * 2, cudaMemcpyDeviceToHost));
        check(memcmp(ref.data(), got.data(), n * 2) == 0, "blas_gemm fallback N=" + std::to_string(badN) + " not bit-exact");
        CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(d_ref)); CK(cudaFree(d_got));
    }
    {
        const size_t n = 48 * 8;
        const std::vector<uint16_t> x = rand_bf(r, 48 * 256 + 8, 1.0f), w = rand_bf(r, 64 * 256, 0.25f);
        uint16_t *dx = to_dev(x), *dw = to_dev(w), *d_bref = to_dev(std::vector<uint16_t>(48 * 64, 0)), *d_bgot = to_dev(std::vector<uint16_t>(48 * 64, 0));
        float *d_ref = to_dev(std::vector<float>(n, 0.0f)), *d_got = to_dev(std::vector<float>(n, 0.0f));
        ink2::gemm_f32(dx, dw, 48, 256, 8, d_ref, 0);
        ink2::gemm_bf16(dx + 2, dw, 48, 256, 64, d_bref, 0);   // +4 bytes (4-byte aligned for ld32, not 16-byte aligned for cuBLAS)
        {
            ink2rt::BlasScope bs(hd);
            check(!ink2rt::blas_gemm(dx, dw, 48, 256, 8, d_got, ink2::GEMM_F32, 0), "blas_gemm F32 N=8 did not fall back");
            check(!ink2rt::blas_gemm(dx + 2, dw, 48, 256, 64, d_bgot, ink2::GEMM_BF16, 0), "blas_gemm unaligned ptr did not fall back");
            ink2::gemm_f32(dx, dw, 48, 256, 8, d_got, 0);
            ink2::gemm_bf16(dx + 2, dw, 48, 256, 64, d_bgot, 0);
        }
        check(ink2::gemm_hook() == nullptr && ink2rt::blas_cur() == nullptr, "BlasScope destructor left hook active");
        CK(cudaDeviceSynchronize());
        std::vector<float> ref(n), got(n);
        std::vector<uint16_t> bref(48 * 64), bgot(48 * 64);
        CK(cudaMemcpy(ref.data(), d_ref, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got.data(), d_got, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bref.data(), d_bref, bref.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bgot.data(), d_bgot, bgot.size() * 2, cudaMemcpyDeviceToHost));
        check(memcmp(ref.data(), got.data(), n * 4) == 0, "blas_gemm F32 N=8 fallback not bit-exact");
        check(memcmp(bref.data(), bgot.data(), bref.size() * 2) == 0, "blas_gemm unaligned ptr fallback not bit-exact");
        CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(d_ref)); CK(cudaFree(d_got)); CK(cudaFree(d_bref)); CK(cudaFree(d_bgot));
    }
    cublasDestroy(hd);
}

static std::unique_ptr<Engine> engine(const std::string& dir, int T, int ghp, int blas, int mtp = 0, int nc = NCAND)
{
    Options o;
    o.nr = T;
    o.rank0 = 0;
    o.nloc = T;
    o.slots = 1;
    o.mtp = mtp;
    o.max_rows = ROWS;
    o.cap = CAP;
    o.nc = nc;
    o.fast_allreduce = 0;
    o.gpu_head_pack = ghp;
    o.cublas_prefill = blas;
    o.tc_prefill_attn = blas;
    std::unique_ptr<Engine> e(new Engine);
    e->init(dir, o);
    return e;
}
static void bench_gemm(int M, int K, int N)
{
    Rng r{0x67656d6du + static_cast<uint64_t>(M)};
    const std::vector<uint16_t> x = rand_bf(r, static_cast<size_t>(M) * K, 1.0f), w = rand_bf(r, static_cast<size_t>(N) * K, 0.2f);
    uint16_t *dx = to_dev(x), *dw = to_dev(w), *dy = to_dev(std::vector<uint16_t>(static_cast<size_t>(M) * N, 0));
    cublasHandle_t hd = nullptr;
    if (cublasCreate(&hd) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "FATAL cublasCreate\n"); exit(2); }
    ink2::gemm(dx, dw, M, K, N, dy, ink2::GEMM_BF16, 0, 0, 0);
    { ink2rt::BlasScope bs(hd); ink2::gemm(dx, dw, M, K, N, dy, ink2::GEMM_BF16, 0, 0, 0); }
    CK(cudaDeviceSynchronize());
    double t0 = now_ms();
    ink2::gemm(dx, dw, M, K, N, dy, ink2::GEMM_BF16, 0, 0, 0);
    CK(cudaDeviceSynchronize());
    const double t_stock = now_ms() - t0;
    t0 = now_ms();
    { ink2rt::BlasScope bs(hd); ink2::gemm(dx, dw, M, K, N, dy, ink2::GEMM_BF16, 0, 0, 0); }
    CK(cudaDeviceSynchronize());
    const double t_blas = now_ms() - t0;
    printf("   bench gemm: %d x %d x %d: stock %.3f ms, cuBLAS %.3f ms\n", M, K, N, t_stock, t_blas);
    cublasDestroy(hd);
    CK(cudaFree(dx));
    CK(cudaFree(dw));
    CK(cudaFree(dy));
}
// every row's output of the teacher-forced run: the prompt in chunks of ROWS, then single rows
static std::vector<RowOut> run(Engine& e, const std::vector<int>& toks, std::vector<int>* drafts = nullptr)
{
    std::vector<RowOut> all, out;
    e.reset_slot(0);
    int p = 0;
    const int P = static_cast<int>(toks.size());
    while (p < P)
    {
        const int n = p < PROMPT ? std::min(ROWS, PROMPT - p) : 1;
        ink2::RowsHost rh;
        rh.add(0, p, n);
        std::vector<RowSample> sm;
        for (int i = 0; i < n; ++i) sm.push_back(RowSample{0, 0, 0, static_cast<uint32_t>(p + i + 1)});
        e.forward(rh, &toks[static_cast<size_t>(p)], sm.data(), out);
        e.commit();
        all.insert(all.end(), out.begin(), out.end());
        p += n;
        if (drafts && p == PROMPT)
            e.mtp_draft({0}, {p - 1}, {out.back().tok}, {RowSample{0, 0, 0, 0}}, 3, *drafts);
    }
    return all;
}
static std::vector<RowOut> run_chunk(Engine& e, const std::vector<int>& toks, int n)
{
    std::vector<RowOut> out;
    e.reset_slot(0);
    ink2::RowsHost rh;
    rh.add(0, 0, n);
    std::vector<RowSample> sm;
    for (int i = 0; i < n; ++i) sm.push_back(RowSample{0, 0, 0, static_cast<uint32_t>(i + 1)});
    e.forward(rh, toks.data(), sm.data(), out);
    e.commit();
    return out;
}
static bool unpack_valid_logits(const RowOut& r, float* dst, const std::string& what)
{
    if (r.top.size() != static_cast<size_t>(VALID))
    {
        check(false, what + ": finite candidate count " + std::to_string(r.top.size()) + " (want " + std::to_string(VALID) + ")");
        return false;
    }
    std::vector<uint8_t> seen(static_cast<size_t>(VALID), 0);
    bool ok = true;
    float mx = -INFINITY;
    for (const ink2rt::Cand& c : r.top)
    {
        if (c.id < 0 || c.id >= VALID || seen[static_cast<size_t>(c.id)] || !std::isfinite(c.logit) || std::fabs(c.logit) > 30.0f)
        {
            ok = false;
            continue;
        }
        seen[static_cast<size_t>(c.id)] = 1;
        dst[c.id] = c.logit;
        mx = std::max(mx, c.logit);
    }
    check(ok, what + ": invalid candidate id or logit");
    if (ok)
    {
        double sum = 0.0;
        for (int v = 0; v < VALID; ++v) sum += std::exp(static_cast<double>(dst[v] - mx));
        const float lse_ref = mx + static_cast<float>(std::log(sum));
        check(std::isfinite(r.lse) && std::fabs(r.lse - lse_ref) <= 1e-5f * std::max(1.0f, std::fabs(lse_ref)), what + ": lse mismatch");
    }
    return ok;
}
struct RowAcc
{
    double dot = 0, na = 0, nb = 0, nd = 0, min_row_cos = 2.0;
    size_t rows = 0;
    void add_row(const float* got, const float* ref, const std::string& what)
    {
        ++rows;
        double rd = 0, ra = 0, rb = 0;
        for (int v = 0; v < VALID; ++v)
        {
            const double a = got[v], b = ref[v];
            rd += a * b;
            ra += a * a;
            rb += b * b;
            nd += (a - b) * (a - b);
        }
        dot += rd;
        na += ra;
        nb += rb;
        if (!(ra > 0 && rb > 0))
        {
            check(false, what + ": degenerate all-zero valid logits");
            return;
        }
        const double rc = rd / std::sqrt(ra * rb);
        min_row_cos = std::min(min_row_cos, rc);
        check(rc >= E2E_ROW_COS, what + ": row cosine " + std::to_string(rc));
    }
    double cos() const { return na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0; }
    double rel() const { return nb > 0 ? std::sqrt(nd / nb) : 1.0; }
};
static void check_exact_rows(const RowOut& a, const RowOut& b, size_t n_top, const std::string& what)
{
    check(a.tok == b.tok, what + ": token " + std::to_string(b.tok) + " vs " + std::to_string(a.tok));
    check(std::isfinite(a.lse) && std::isfinite(b.lse) && std::fabs(a.lse - b.lse) <= 1e-5f * std::max(1.0f, std::fabs(a.lse)), what + ": lse");
    check(a.top.size() >= n_top && b.top.size() >= n_top && (n_top == 0 || memcmp(a.top.data(), b.top.data(), n_top * sizeof(ink2rt::Cand)) == 0),
          what + ": top-" + std::to_string(n_top) + " candidates");
}
static void check_engine(const std::string& dir, int T, const std::vector<int>& toks)
{
    const std::string tag = "engine of " + std::to_string(T) + " rank(s)";
    std::unique_ptr<Engine> e_off64 = engine(dir, T, 0, 0, 3, NCAND), e_ghp64 = engine(dir, T, 1, 0, 3, NCAND), e_all64 = engine(dir, T, 1, 1, 3, NCAND),
                            e_off_v = engine(dir, T, 0, 0, 3, VOCAB), e_all_v = engine(dir, T, 0, 1, 3, VOCAB);
    check(!e_off64->gpu_head_pack_active() && !e_off64->cublas_prefill_active(), tag + ": off64 flags");
    check(e_ghp64->gpu_head_pack_active() && !e_ghp64->cublas_prefill_active(), tag + ": ghp64 flags");
    check(e_all64->gpu_head_pack_active() && e_all64->cublas_prefill_active(), tag + ": all64 flags");
    check(!e_off_v->gpu_head_pack_active() && !e_off_v->cublas_prefill_active(), tag + ": off_v flags");
    check(!e_all_v->gpu_head_pack_active() && e_all_v->cublas_prefill_active(), tag + ": all_v flags");

    // 3a. M = 24 < 32 prefill chunk: BlasScope stays off, so all64 is bit-exact against stock off64.
    const std::vector<RowOut> c24_off = run_chunk(*e_off64, toks, 24), c24_all = run_chunk(*e_all64, toks, 24);
    check(c24_off.size() == 24 && c24_all.size() == 24, tag + ": 24-row count");
    for (size_t i = 0; i < std::min(c24_off.size(), c24_all.size()); ++i)
        check_exact_rows(c24_off[i], c24_all[i], static_cast<size_t>(NCAND), tag + ", M=24 row " + std::to_string(i));

    // 3b. Full 48 + 22 + 8x1 = 78 rows + 3-draft MTP round:
    std::vector<int> d_off64, d_ghp64, d_all64, d_off_v, d_all_v;
    const std::vector<RowOut> off64 = run(*e_off64, toks, &d_off64), ghp64 = run(*e_ghp64, toks, &d_ghp64), all64 = run(*e_all64, toks, &d_all64),
                              off_v = run(*e_off_v, toks, &d_off_v), all_v = run(*e_all_v, toks, &d_all_v);
    const size_t N = toks.size();
    check(off64.size() == N && ghp64.size() == N && all64.size() == N && off_v.size() == N && all_v.size() == N, tag + ": row count");
    check(d_off64.size() == 3 && d_off64 == d_ghp64 && d_off64 == d_off_v, tag + ": mtp_draft stock / device head pack / full vocab");
    check(d_all64.size() == 3 && d_all64 == d_all_v, tag + ": mtp_draft all64 vs all_v");
    for (int t : d_all64) check(t >= 0 && t < VALID, tag + ": mtp_draft all64 invalid token " + std::to_string(t));
    if (off64.size() != N || ghp64.size() != N || all64.size() != N || off_v.size() != N || all_v.size() != N) return;

    RowAcc pf_blas, pf_tail, dec_rows;
    std::vector<float> lg_off(static_cast<size_t>(VALID), 0.0f), lg_all(static_cast<size_t>(VALID), 0.0f);
    size_t same_tok = 0;
    for (size_t i = 0; i < N; ++i)
    {
        const std::string wi = tag + ", row " + std::to_string(i);
        check(off64[i].top.size() == static_cast<size_t>(NCAND) && ghp64[i].top.size() == static_cast<size_t>(NCAND) && all64[i].top.size() == static_cast<size_t>(NCAND),
              wi + ": top-64 count");
        check_exact_rows(off64[i], ghp64[i], static_cast<size_t>(NCAND), wi + " (ghp64 vs off64)");
        check_exact_rows(off_v[i], off64[i], static_cast<size_t>(NCAND), wi + " (off64 vs off_v)");
        check_exact_rows(all_v[i], all64[i], static_cast<size_t>(NCAND), wi + " (all64 vs all_v)");
        if (all_v[i].tok == off_v[i].tok) ++same_tok;
        const bool ok_o = unpack_valid_logits(off_v[i], lg_off.data(), wi + " stock"), ok_a = unpack_valid_logits(all_v[i], lg_all.data(), wi + " cuBLAS");
        if (!ok_o || !ok_a) continue;
        if (i < static_cast<size_t>(ROWS)) pf_blas.add_row(lg_all.data(), lg_off.data(), wi + " pf48");
        else if (i < static_cast<size_t>(PROMPT)) pf_tail.add_row(lg_all.data(), lg_off.data(), wi + " pf22");
        else dec_rows.add_row(lg_all.data(), lg_off.data(), wi + " dec8");
    }
    const auto chk_acc = [&](const RowAcc& a, size_t want_rows, const char* seg) {
        const std::string s = tag + " " + seg;
        check(a.rows == want_rows, s + ": rows " + std::to_string(a.rows));
        check(a.cos() >= E2E_SEQ_COS, s + ": cosine " + std::to_string(a.cos()));
        check(a.rel() <= E2E_SEQ_REL, s + ": relative " + std::to_string(a.rel()));
    };
    chk_acc(pf_blas, static_cast<size_t>(ROWS), "pf48");
    chk_acc(pf_tail, static_cast<size_t>(PROMPT - ROWS), "pf22");
    chk_acc(dec_rows, static_cast<size_t>(SINGLES), "dec8");
    const double min_rc = std::min({pf_blas.min_row_cos, pf_tail.min_row_cos, dec_rows.min_row_cos});
    printf("   %d rank(s): %zu rows; ghp64 & M=24 exact; cuBLAS %zu/%zu same argmax | pf48 cos %.5f rel %.4f | pf22 cos %.5f rel %.4f | dec8 cos %.5f rel %.4f | min row cos %.5f\n",
           T, N, same_tok, N, pf_blas.cos(), pf_blas.rel(), pf_tail.cos(), pf_tail.rel(), dec_rows.cos(), dec_rows.rel(), min_rc);
}

// ------------------------------------------------------------------------------------- 4. the device all-reduce
// A fabric of 4 ranks in pinned host memory with no proxy: this test is rank 1 and plays the three peers. Even cases
// have the peers' shares and tail words in the receive slots before the launches; in odd cases the host waits for the
// doorbell, then writes the shares and, last, the tail words, while the device spins.
template <class T> static T* pinned(size_t n)
{
    void* p = nullptr;
    CK(cudaHostAlloc(&p, std::max<size_t>(1, n) * sizeof(T), cudaHostAllocMapped));
    memset(p, 0, std::max<size_t>(1, n) * sizeof(T));
    return static_cast<T*>(p);
}
static void check_fabric()
{
    const int NR = 4, ME = 1, NCOLL = 96;
    const size_t max_bytes = static_cast<size_t>(512) * 4096 * 4;
    g6::FabricConfig cfg;
    cfg.slot_bytes = max_bytes;
    cfg.pool = 2;
    cfg.run1 = cfg.run2 = NCOLL;
    cfg.slot_bytes1 = 8;
    g6::FabricDev f{};
    f.rank = ME;
    f.nr = NR;
    if (!g6::fab_slot_layout(cfg, NCOLL, &f)) { check(false, "fabric: the pooled layout is refused"); return; }
    const size_t area = g6::fab_slot_base(f, NCOLL);
    check(f.pool == 2 && area == 2 * f.stride && g6::fab_recv_bytes(f) == 2 * area * NR, "fabric: the pooled areas");
    bool lay = true;
    for (int par = 0; par < 2; ++par)
        for (int c = 0; c < NCOLL; ++c)
        {
            lay = lay && g6::fab_pool_slot(f, c) == c % 2 && g6::fab_send_off(f, par, c) == par * area + (c % 2) * f.stride;
            for (int p = 0; p < NR; ++p) lay = lay && g6::fab_recv_off(f, par, c, p) == (par * area + (c % 2) * f.stride) * NR + p * f.stride;
        }
    check(lay, "fabric: the slot offsets of the pooled layout");
    std::vector<int> walk(NCOLL), credit;
    for (int c = 0; c < NCOLL; ++c) walk[c] = c;
    bool cr = g6::fab_pool_credits(f, walk, &credit) && credit[0] == -1 && credit[1] == -1;
    for (int c = 2; cr && c < NCOLL; ++c) cr = credit[c] == c - 1;
    check(cr, "fabric: the pool credits of a step");

    f.send = pinned<char>(2 * area);
    f.recv = pinned<char>(g6::fab_recv_bytes(f));
    f.doorbell = pinned<unsigned long long>(NCOLL);
    f.error = pinned<unsigned>(1);
    Rng r{0x66616272u};
    int k = 0;
    for (int M : {1, 8, 64, 512})
        for (int cols : {2304, 4096})
        {
            const size_t n = static_cast<size_t>(M) * cols, bytes = n * 4;
            const unsigned long long step = 3 + k;
            const int c = (k * 13 + 5) % NCOLL, par = static_cast<int>(step & 1);
            const unsigned long long seq = g6::coll_seq(step, c, NCOLL);
            const std::string tag = "fabric: M " + std::to_string(M) + " cols " + std::to_string(cols) + ": ";
            std::vector<float> parts(NR * n), mine(NR * n);
            for (float& x : parts) x = unit(r) * 3.0f;
            memset(mine.data(), 0xff, mine.size() * 4);   // the peers' places hold NaN until the exchange fills them
            memcpy(mine.data() + ME * n, parts.data() + ME * n, bytes);
            float* d_ref = to_dev(parts);
            float* d_mine = to_dev(mine);
            uint16_t *o_ref = nullptr, *o_mine = nullptr;
            CK(cudaMalloc(&o_ref, n * 2));
            CK(cudaMalloc(&o_mine, n * 2));
            ink2::sum_ranks(d_ref, NR, n, o_ref, 0);
            CK(cudaStreamSynchronize(0));
            auto peers = [&](bool tails) {
                for (int p = 0; p < NR; ++p)
                {
                    if (p == ME) continue;
                    char* slot = f.recv + g6::fab_recv_off(f, par, c, p);
                    if (!tails) memcpy(slot, parts.data() + p * n, bytes);
                    else __atomic_store_n(reinterpret_cast<unsigned long long*>(slot + bytes), seq, __ATOMIC_SEQ_CST);
                }
            };
            const bool late = k & 1;
            if (!late) { peers(false); peers(true); }
            ink2::ar_exchange(f, step, c, reinterpret_cast<char*>(d_mine), bytes, 0);
            ink2::sum_ranks(d_mine, NR, n, o_mine, 0);
            if (late)
            {
                const double t0 = now_ms();
                while (__atomic_load_n(f.doorbell + c, __ATOMIC_SEQ_CST) != seq && now_ms() - t0 < 20000.0) {}
                check(__atomic_load_n(f.doorbell + c, __ATOMIC_SEQ_CST) == seq, tag + "no doorbell in 20 s");
                peers(false);
                peers(true);
            }
            CK(cudaStreamSynchronize(0));
            std::vector<uint16_t> a(n), b(n);
            CK(cudaMemcpy(a.data(), o_ref, n * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), o_mine, n * 2, cudaMemcpyDeviceToHost));
            check(memcmp(a.data(), b.data(), n * 2) == 0, tag + "the sum is not the stock sum's bits");
            check(f.doorbell[c] == seq, tag + "the doorbell word");
            check(memcmp(f.send + g6::fab_send_off(f, par, c), parts.data() + ME * n, bytes) == 0, tag + "the send slot");
            check(*f.error == 0, tag + "the error word is set");
            CK(cudaFree(d_ref));
            CK(cudaFree(d_mine));
            CK(cudaFree(o_ref));
            CK(cudaFree(o_mine));
            ++k;
        }
    printf("   %d collectives of 4 ranks: the sum's bits are the stock sum's\n", k);
    CK(cudaFreeHost(f.send));
    CK(cudaFreeHost(f.recv));
    CK(cudaFreeHost(f.doorbell));
    CK(cudaFreeHost(f.error));
}

int main(int argc, char** argv)
{
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    knob::init();
    printf("1. the head's candidates packed on the device, and BlasScope / blas_gemm\n");
    for (int nc : {16, 256, 512})
    {
        check_head_pack(1000, nc, 9, false);
        check_head_pack(12576, nc, 9, false);
    }
    check_head_pack(255, 256, 5, false);
    check_blas_gemm();
    if (bench)
    {
        check_head_pack(50304, 256, 8, true);
        bench_gemm(128, 4608, 6144);
        bench_gemm(512, 4608, 6144);
    }

    printf("2. the tensor-core attention of a prefill chunk\n");
    check_attn_tc({"local, ring and chunk", 40, 200, 12, 0, 0, 2, 96, 0}, false);
    check_attn_tc({"local, split K / V head", 40, 37, 9, 3, 0, 2, 96, 0}, false);
    check_attn_tc({"local, first positions", 40, 0, 12, 0, 0, 2, 96, 0}, false);
    check_attn_tc({"global, pages", 40, 100, 12, 6, 1, 2, 96, 1}, false);
    check_attn_tc({"global, past the rel window", 24, 1100, 6, 0, 0, 1, 96, 1}, false);
    if (bench)
    {
        check_attn_tc({"local, ctx 2048, window 1024", 512, 2048, 12, 0, 0, 2, 1024, 0}, true);
        check_attn_tc({"global, ctx 8192", 512, 7680, 12, 0, 0, 2, 1024, 1}, true);
    }

    printf("3. the engine with the speed paths on and off\n");
    Mini mm;
    build_mini(mm);
    char tmpl[] = "/tmp/ink2-speed-test.XXXXXX";
    if (!mkdtemp(tmpl)) { perror("mkdtemp"); return 2; }
    const std::string tmp = tmpl, dir = tmp + "/ckpt";
    write_ckpt(mm, dir);
    Rng r{0x73706565u};
    std::vector<int> toks;
    for (int i = 0; i < PROMPT + SINGLES; ++i) toks.push_back(static_cast<int>(r.next() % VALID));
    for (int T : {1, 4, 6}) check_engine(dir, T, toks);
    std::filesystem::remove_all(tmp);

    printf("4. the device all-reduce on a fabric in memory\n");
    check_fabric();

    if (fails)
    {
        printf("FAIL (%d)\n", fails);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
