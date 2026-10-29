// GR8_GLM53F_PF_DSA_F16GEMM unit test: MXFP8 GV records -> fp16 matrix + cuBLAS, and two BF16W records -> one stacked bf16
// matrix + cuBLAS (flash_dsa_f16.cuh), against the full-GV kernel on the same operands and both against fp64.
//   cd engine/runtime && nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a <MODEL=flash build flags> -I.. \
//       tests/flash_dsa_f16_test.cu -o flash_dsa_f16_test -lcublas
//   ./flash_dsa_f16_test --self-test   host only: MXFP8 and BF16W record offset maps (no CUDA calls)
//   ./flash_dsa_f16_test               GPU: q_a, kv_a, idx_k, idx_q (MXFP8) and idx_w + idx_g (BF16W) at the engine
//                                      shapes, M 8192 and 1000; idx_k with a large row 0 at M 255; inexact fallback
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "runtime/rank.cuh"
#include "runtime/flash_dsa_pf.cuh"

using namespace g6rt;

static int self_test()
{
    int fails = 0;
    for (auto [O, K] : {std::pair{32, 64}, std::pair{1536, 4096}, std::pair{4096, 1536}})
    {
        const int NG = K / 32;
        const size_t bytes = static_cast<size_t>(O / 16) * NG * 528;
        std::vector<int> hit(bytes, 0);
        for (int o = 0; o < O; ++o)
            for (int k = 0; k < K; ++k)
            {
                const size_t c = flash_dsa_f16::mx_code_offset(o, k, NG), s = flash_dsa_f16::mx_scale_offset(o, k, NG);
                if (c >= bytes || c % 528 >= 512 || s >= bytes || s % 528 < 512) { ++fails; continue; }
                ++hit[c];
                if (s / 528 != c / 528) ++fails;   // the scale lives in the code's group record
            }
        size_t codes = 0, bad = 0;
        for (size_t i = 0; i < bytes; ++i)
            if (i % 528 < 512) { codes++; if (hit[i] != 1) bad++; }
        // lane mapping spot checks against the densev.cuh comment: lane 5 (r 1, c 1) word 0 = {row 1 k 2,3 | row 9 k 2,3}
        if (flash_dsa_f16::mx_code_offset(1, 2, NG) != 5 * 8 + 0 || flash_dsa_f16::mx_code_offset(9, 3, NG) != 5 * 8 + 3 ||
            flash_dsa_f16::mx_code_offset(1, 10, NG) != 5 * 8 + 4 || flash_dsa_f16::mx_code_offset(0, 16, NG) != 256 ||
            flash_dsa_f16::mx_scale_offset(9, 0, NG) != 512 + 3)
            ++fails;
        std::printf("SELF O=%d K=%d code bytes %zu, not hit exactly once %zu\n", O, K, codes, bad);
        fails += static_cast<int>(bad);
    }
    for (auto [O, K] : {std::pair{16, 16}, std::pair{32, 4096}, std::pair{128, 4096}})
    {
        const int NG = K / 16;
        const size_t bytes = static_cast<size_t>(O / 16) * NG * 512;
        std::vector<int> hit(bytes, 0);
        for (int o = 0; o < O; ++o)
            for (int k = 0; k < K; ++k)
            {
                const size_t b = flash_dsa_f16::bf_offset(o, k, NG);
                if (b >= bytes || b % 2) { ++fails; continue; }
                ++hit[b];
            }
        size_t bad = 0;
        for (size_t i = 0; i < bytes; i += 2) bad += hit[i] != 1;
        // lane 5 (r 1, c 1): a[0] = row 1 k 2,3; a[1] = row 9 k 2,3; a[2] = row 1 k 10,11; a[3] = row 9 k 10,11
        if (flash_dsa_f16::bf_offset(1, 2, NG) != 5 * 16 + 0 || flash_dsa_f16::bf_offset(9, 3, NG) != 5 * 16 + 4 + 2 ||
            flash_dsa_f16::bf_offset(1, 10, NG) != 5 * 16 + 8 || flash_dsa_f16::bf_offset(9, 11, NG) != 5 * 16 + 12 + 2 ||
            (NG > 1 && flash_dsa_f16::bf_offset(0, 16, NG) != 512) || (O > 16 && flash_dsa_f16::bf_offset(16, 0, NG) != static_cast<size_t>(NG) * 512))
            ++fails;
        std::printf("SELF bf16 O=%d K=%d elements %zu, not hit exactly once %zu\n", O, K, bytes / 2, bad);
        fails += static_cast<int>(bad);
    }
    std::printf(fails ? "SELF-TEST FAIL %d\n" : "SELF-TEST OK%.0d device_calls=0\n", fails);
    return fails ? 1 : 0;
}

static float e4m3(uint8_t b)
{
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
    return s ? -v : v;
}
static float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// row0: row 0's activations scaled by this, as the first token's are in the model
struct Case { const char* name; int O, K, tpc, ks, M, spread; bool expect_exact; float row0 = 1.f; };

static int run(const Case& c, cublasHandle_t, cudaStream_t st, std::mt19937& rng)
{
    const int NG = c.K / 32;
    const size_t rb = static_cast<size_t>(c.O / 16) * NG * 528;
    std::vector<uint8_t> rec(rb);
    std::uniform_int_distribution<int> byte(0, 255), sp(0, c.spread);
    for (size_t i = 0; i < rb; ++i)
    {
        if (i % 528 >= 512) rec[i] = static_cast<uint8_t>(115 - sp(rng));   // scales 2^-12 .. 2^(-12-spread)
        else { int b; do b = byte(rng); while ((b & 0x7f) == 0x7f); rec[i] = static_cast<uint8_t>(b); }
    }
    // activations: fp32 values that are exact fp16 numbers, so the f16 fragments and the plain fp16 rows hold the same
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x32(static_cast<size_t>(c.M) * c.K);
    std::vector<__half> x16(x32.size());
    for (size_t i = 0; i < x32.size(); ++i)
    {
        x16[i] = __float2half_rn(nd(rng) * (i < static_cast<size_t>(c.K) ? c.row0 : 1.f));
        x32[i] = __half2float(x16[i]);
    }

    uint8_t* d_rec = dmalloc<uint8_t>(rb);
    float* d_x32 = dmalloc<float>(static_cast<size_t>((c.M + 7) / 8 * 8) * c.K);
    __half* d_x16 = dmalloc<__half>(x16.size());
    u32* d_xf = dmalloc<u32>(flash_dsa_full_schedule::full_fragment_words(c.M, c.K));
    __half* d_ygv = dmalloc<__half>(static_cast<size_t>(c.M) * c.O);
    __half* d_yf = dmalloc<__half>(static_cast<size_t>(c.M) * c.O);
    float* d_y32 = dmalloc<float>(static_cast<size_t>(c.M) * c.O);
    G6RT_CK(cudaMemcpy(d_rec, rec.data(), rb, cudaMemcpyHostToDevice));
    G6RT_CK(cudaMemcpy(d_x32, x32.data(), x32.size() * 4, cudaMemcpyHostToDevice));
    G6RT_CK(cudaMemcpy(d_x16, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice));
    flash_mhc::fragments(d_x32, c.M, c.K, d_xf, nullptr, st);

    bind::Gv g;
    // the bind::make_gv split rule: ks halves until it divides the group count, tpc * ks <= 32 (idx_q: ks 32 -> 16)
    int ks = c.ks, tpc = c.tpc;
    while (ks > 1 && NG % ks) ks >>= 1;
    while (tpc * ks > 32) tpc >>= 1;
    g.on = 1; g.codec = bind::MXFP8; g.w = d_rec; g.K = c.K; g.O = c.O; g.tpc = tpc; g.ks = ks; g.ld = c.O; g.odt = gv::O_F16;
    flash_dsa_pf::launch_gv_full(g, d_xf, c.M, d_ygv, st);
    flash_dsa_f16::Mat w;
    const bool built = flash_dsa_f16::build(g, w, c.name, 0, st);
    if (built != c.expect_exact)
    {
        std::printf("%s M=%d: build %s, expected %s FAIL\n", c.name, c.M, built ? "exact" : "fallback", c.expect_exact ? "exact" : "fallback");
        return 1;
    }
    if (!built) { std::printf("%s M=%d spread=%d: fallback as expected OK\n", c.name, c.M, c.spread); return 0; }
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) die("cublasCreate");
    flash_dsa_f16::gemm(h, w, d_x16, c.K, c.M, d_y32, d_yf, c.O, st);
    G6RT_CK(cudaStreamSynchronize(st));
    std::vector<uint16_t> ygv(static_cast<size_t>(c.M) * c.O), yf(ygv.size());
    G6RT_CK(cudaMemcpy(ygv.data(), d_ygv, ygv.size() * 2, cudaMemcpyDeviceToHost));
    G6RT_CK(cudaMemcpy(yf.data(), d_yf, yf.size() * 2, cudaMemcpyDeviceToHost));

    size_t equal = 0, ulp1 = 0, worse = 0;
    for (size_t i = 0; i < ygv.size(); ++i)
    {
        if (ygv[i] == yf[i]) { ++equal; continue; }
        const int a = static_cast<int16_t>(ygv[i]) < 0 ? -(ygv[i] & 0x7fff) : ygv[i];
        const int b = static_cast<int16_t>(yf[i]) < 0 ? -(yf[i] & 0x7fff) : yf[i];
        if (std::abs(a - b) <= 1) ++ulp1; else ++worse;
    }
    // fp64 reference on 64 rows spread over M
    double egv = 0, ef = 0, mag = 0;
    for (int t = 0; t < 64; ++t)
    {
        const int m = static_cast<int>((static_cast<long long>(t) * (c.M - 1)) / 63);
        for (int o = 0; o < c.O; ++o)
        {
            double acc = 0;
            for (int k = 0; k < c.K; ++k)
            {
                const double s = std::ldexp(1.0, rec[flash_dsa_f16::mx_scale_offset(o, k, NG)] - 127);
                acc += static_cast<double>(e4m3(rec[flash_dsa_f16::mx_code_offset(o, k, NG)])) * s * x32[static_cast<size_t>(m) * c.K + k];
            }
            const size_t i = static_cast<size_t>(m) * c.O + o;
            egv = std::max(egv, std::fabs(h2f(ygv[i]) - acc));
            ef = std::max(ef, std::fabs(h2f(yf[i]) - acc));
            mag = std::max(mag, std::fabs(acc));
        }
    }
    const double n = static_cast<double>(ygv.size());
    // gate: f16 no less accurate than GV against fp64; the f16 sum order differs, so >1-ulp vs GV is reported, not gated
    const bool ok = ef <= egv * 1.5 + 1e-30;
    std::printf("%s O=%d K=%d tpc=%d ks=%d M=%d alpha=2^%d: equal %.4f, 1-ulp %.4f, >1-ulp %zu; fp64 ref max abs err gv %.3g f16 %.3g (max |y| %.3g) %s\n",
                c.name, c.O, c.K, tpc, ks, c.M, static_cast<int>(std::log2(w.alpha)), equal / n, ulp1 / n, worse, egv, ef, mag, ok ? "OK" : "FAIL");
    cublasDestroy(h);
    cudaFree(w.w); cudaFree(d_rec); cudaFree(d_x32); cudaFree(d_x16); cudaFree(d_xf); cudaFree(d_ygv); cudaFree(d_yf); cudaFree(d_y32);
    return ok ? 0 : 1;
}

static float bf2f(uint16_t b) { uint32_t u = static_cast<uint32_t>(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

// idx_w (Oa x K) and idx_g (Ob x K) BF16W records: GV each vs one stacked bf16 cuBLAS GEMM, split to fp16
static int run_bf(const char* name, int Oa, int Ob, int K, int M, std::mt19937& rng, cudaStream_t st)
{
    const int NG = K / 16;
    std::normal_distribution<float> nd(0.f, 1.f);
    auto record = [&](int O) {
        std::vector<uint8_t> rec(static_cast<size_t>(O / 16) * NG * 512);
        for (size_t i = 0; i < rec.size(); i += 2)
        {
            const __nv_bfloat16 b = __float2bfloat16_rn(nd(rng) * 0.02f);
            const uint16_t u = __bfloat16_as_ushort(b);
            rec[i] = static_cast<uint8_t>(u & 0xff); rec[i + 1] = static_cast<uint8_t>(u >> 8);
        }
        return rec;
    };
    const std::vector<uint8_t> ra = record(Oa), rb = record(Ob);
    std::vector<float> x32(static_cast<size_t>(M) * K);
    for (float& v : x32) v = nd(rng);
    uint8_t* d_ra = dmalloc<uint8_t>(ra.size());
    uint8_t* d_rb = dmalloc<uint8_t>(rb.size());
    float* d_x32 = dmalloc<float>(static_cast<size_t>((M + 7) / 8 * 8) * K);
    u32* d_xf = dmalloc<u32>(flash_dsa_full_schedule::full_fragment_words(M, K));
    u32* d_xb = dmalloc<u32>(flash_dsa_full_schedule::full_fragment_words(M, K));
    __half *d_ga = dmalloc<__half>(static_cast<size_t>(M) * Oa), *d_gb = dmalloc<__half>(static_cast<size_t>(M) * Ob);
    __half *d_fa = dmalloc<__half>(static_cast<size_t>(M) * Oa), *d_fb = dmalloc<__half>(static_cast<size_t>(M) * Ob);
    __nv_bfloat16* d_rows = dmalloc<__nv_bfloat16>(static_cast<size_t>(M) * K);
    float* d_y32 = dmalloc<float>(static_cast<size_t>(M) * (Oa + Ob));
    G6RT_CK(cudaMemcpy(d_ra, ra.data(), ra.size(), cudaMemcpyHostToDevice));
    G6RT_CK(cudaMemcpy(d_rb, rb.data(), rb.size(), cudaMemcpyHostToDevice));
    G6RT_CK(cudaMemcpy(d_x32, x32.data(), x32.size() * 4, cudaMemcpyHostToDevice));
    flash_mhc::fragments(d_x32, M, K, d_xf, d_xb, st);
    auto gv = [&](const uint8_t* rec, int O, __half* y) {
        bind::Gv g;
        int ks = 32, tpc = 1;   // engine idx_w / idx_g split (flash_dsa.cuh), make_gv rule
        while (ks > 1 && NG % ks) ks >>= 1;
        while (tpc * ks > 32) tpc >>= 1;
        g.on = 1; g.codec = bind::BF16W; g.w = rec; g.K = K; g.O = O; g.tpc = tpc; g.ks = ks; g.ld = O; g.odt = gv::O_F16;
        flash_dsa_pf::launch_gv_full(g, d_xb, M, y, st);
        return g;
    };
    const bind::Gv ga = gv(d_ra, Oa, d_ga), gb = gv(d_rb, Ob, d_gb);
    flash_dsa_f16::Mat w;
    if (!flash_dsa_f16::build_bf_cat(ga, gb, w, st)) { std::printf("%s: build_bf_cat refused FAIL\n", name); return 1; }
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) die("cublasCreate");
    flash_dsa_f16::gemm_bf_split(h, w, d_xb, M, d_rows, d_y32, d_fa, Oa, d_fb, Ob, st);
    G6RT_CK(cudaStreamSynchronize(st));
    G6RT_CK(cudaGetLastError());
    int fails = 0;
    for (int part = 0; part < 2; ++part)
    {
        const int O = part ? Ob : Oa;
        const std::vector<uint8_t>& rec = part ? rb : ra;
        std::vector<uint16_t> yg(static_cast<size_t>(M) * O), yf(yg.size());
        G6RT_CK(cudaMemcpy(yg.data(), part ? d_gb : d_ga, yg.size() * 2, cudaMemcpyDeviceToHost));
        G6RT_CK(cudaMemcpy(yf.data(), part ? d_fb : d_fa, yf.size() * 2, cudaMemcpyDeviceToHost));
        size_t equal = 0, ulp1 = 0, worse = 0;
        for (size_t i = 0; i < yg.size(); ++i)
        {
            if (yg[i] == yf[i]) { ++equal; continue; }
            const int a = static_cast<int16_t>(yg[i]) < 0 ? -(yg[i] & 0x7fff) : yg[i];
            const int b = static_cast<int16_t>(yf[i]) < 0 ? -(yf[i] & 0x7fff) : yf[i];
            if (std::abs(a - b) <= 1) ++ulp1; else ++worse;
        }
        double egv = 0, ef = 0, mag = 0;
        for (int t = 0; t < 64; ++t)
        {
            const int m = static_cast<int>((static_cast<long long>(t) * (M - 1)) / 63);
            for (int o = 0; o < O; ++o)
            {
                double acc = 0;
                for (int k = 0; k < K; ++k)
                {
                    const size_t b = flash_dsa_f16::bf_offset(o, k, NG);
                    const double wv = bf2f(static_cast<uint16_t>(rec[b] | (rec[b + 1] << 8)));
                    acc += wv * static_cast<double>(__bfloat162float(__float2bfloat16_rn(x32[static_cast<size_t>(m) * K + k])));
                }
                const size_t i = static_cast<size_t>(m) * O + o;
                egv = std::max(egv, std::fabs(h2f(yg[i]) - acc));
                ef = std::max(ef, std::fabs(h2f(yf[i]) - acc));
                mag = std::max(mag, std::fabs(acc));
            }
        }
        const double n = static_cast<double>(yg.size());
        const bool ok = ef <= egv * 1.5 + 1e-30;
        std::printf("%s.%s O=%d K=%d M=%d (stacked %d): equal %.4f, 1-ulp %.4f, >1-ulp %zu; fp64 ref max abs err gv %.3g bf %.3g (max |y| %.3g) %s\n",
                    name, part ? "idx_g" : "idx_w", O, K, M, w.O, equal / n, ulp1 / n, worse, egv, ef, mag, ok ? "OK" : "FAIL");
        fails += !ok;
    }
    cublasDestroy(h);
    for (void* q : {w.w, static_cast<void*>(d_ra), static_cast<void*>(d_rb), static_cast<void*>(d_x32), static_cast<void*>(d_xf),
                    static_cast<void*>(d_xb), static_cast<void*>(d_ga), static_cast<void*>(d_gb), static_cast<void*>(d_fa),
                    static_cast<void*>(d_fb), static_cast<void*>(d_rows), static_cast<void*>(d_y32)})
        cudaFree(q);
    return fails;
}

int main(int argc, char** argv)
{
    if (argc == 2 && !std::strcmp(argv[1], "--self-test")) return self_test();
    if (self_test()) return 1;
    cudaStream_t st;
    G6RT_CK(cudaStreamCreate(&st));
    std::mt19937 rng(20261016);
    const Case cases[] = {
        {"q_a", 1536, 4096, 2, 16, 8192, 12, true},
        {"q_a", 1536, 4096, 2, 16, 1000, 12, true},
        {"idx_q", 4096, 1536, 1, 32, 8192, 12, true},
        {"idx_q", 4096, 1536, 1, 32, 1000, 12, true},
        {"kv_a", 576, 4096, 1, 32, 8192, 12, true},
        {"kv_a", 576, 4096, 1, 32, 1000, 12, true},
        {"idx_k", 128, 4096, 1, 32, 8192, 12, true},
        {"idx_k", 128, 4096, 1, 32, 1000, 12, true},
        // a large row 0 at a row count where cuBLAS splits K: the unscaled partial sums leave the fp16 range
        {"idx_k_row0", 128, 4096, 1, 32, 255, 12, true, 1024.f},
        {"idx_k_row0", 128, 4096, 1, 32, 1000, 12, true, 1024.f},
        {"q_a_wide_spread", 1536, 4096, 2, 16, 64, 40, false},
    };
    int fails = 0;
    for (const Case& c : cases) fails += run(c, nullptr, st, rng);
    fails += run_bf("wg", 32, 128, 4096, 8192, rng, st);
    fails += run_bf("wg", 32, 128, 4096, 1000, rng, st);
    std::printf(fails ? "FLASH_DSA_F16 FAIL %d\n" : "FLASH_DSA_F16 PASS\n", fails);
    return fails ? 1 : 0;
}
