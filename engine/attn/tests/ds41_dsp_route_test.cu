// GR8_DS_DSP_GV route test (host only, no GPU): the one choke point ds41dsp::dsp_mm that the 19 DSpark projection launches of a step go through.
//   knob off                 -> ds41a::launch_gemm, no GEMV hook runs
//   knob on, 1..8 rows, gv   -> xfrag (unless gv_ready) + gemv hooks, no GEMM
//   knob on, 9 rows / no gv / hooks unbound / g_use_gv off -> the loud-failure hook (never a silent launch_gemm)
//   knob on, prefill launch (wide = true, ds_main_rows): > 8 rows -> the GEMM; 1..8 rows still the GEMV; no gv records / hooks unbound at <= 8 rows still loud
//   GR8_DS_DSP_TC (ds41_gemm_mx_tc): knob off -> never called; knob on -> called only for a wide launch of > 8 rows in one group, with the Fp8Mat's N / K
//   and the caller's strides; rc 0 -> no scalar launch; rc 1 / 2 -> the scalar launch runs (nothing is lost); entry absent (-DROUTE_NO_TC_STUB) -> scalar.
// The hooks are counting stubs; the GEMM branch is the one that runs neither (its kernel launch fails harmlessly: there is no device. Run with
// CUDA_VISIBLE_DEVICES= ; the test exits early when it sees a device, so it can never touch a GPU).
// Also: MainS::carve stays inside MainS::bytes, and the three DSpark functions hold no direct launch_gemm call (checked by the script line below).
//   build: nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I engine -o /tmp/ds41_dsp_route_test engine/attn/tests/ds41_dsp_route_test.cu \
//            engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm
//   run:   CUDA_VISIBLE_DEVICES= /tmp/ds41_dsp_route_test
#include <cstdio>
#include <cstdlib>
#include "../ds41_attn_dsp.cuh"

static int g_xf_calls = 0, g_gv_calls = 0, g_fail_calls = 0, g_fails = 0;
static void stub_fail(const ds41a::Fp8Mat&, int) { ++g_fail_calls; }
static void stub_xfrag(const uint8_t*, int, const uint8_t*, int, int, int, int, size_t, size_t, uint32_t*, cudaStream_t) { ++g_xf_calls; }
static void stub_gemv(const uint32_t*, const ds41a::Fp8Mat&, ds41a::bf16*, int, int, int, size_t, cudaStream_t, int, int) { ++g_gv_calls; }

#ifndef ROUTE_NO_TC_STUB
static int g_tc_calls = 0, g_tc_rc = 0, g_tc_M, g_tc_N, g_tc_K, g_tc_lda, g_tc_lsa, g_tc_ldy;
extern "C" int ds41_gemm_mx_tc(const uint8_t*, int lda, const uint8_t*, int lsa, const uint8_t*, const uint8_t*, void*, int ldy, int M, int N, int K, void*)
{
    ++g_tc_calls; g_tc_M = M; g_tc_N = N; g_tc_K = K; g_tc_lda = lda; g_tc_lsa = lsa; g_tc_ldy = ldy;
    return g_tc_rc;
}
#else
static int g_tc_calls = 0, g_tc_rc = 0, g_tc_M, g_tc_N, g_tc_K, g_tc_lda, g_tc_lsa, g_tc_ldy;
#endif

#define EXPECT(c, ...)                                                       \
    do                                                                       \
    {                                                                        \
        if (!(c))                                                            \
        {                                                                    \
            ++g_fails;                                                       \
            std::printf("FAIL %s:%d  %s  ", __FILE__, __LINE__, #c);        \
            std::printf(__VA_ARGS__);                                        \
            std::printf("\n");                                               \
        }                                                                    \
    } while (0)

// one dsp_mm call; returns 1 when it took the GEMM (neither hook nor the loud-failure hook ran: dsp_mm has exactly these three branches; the launch
// itself fails harmlessly with no device), 0 when it went through the GEMV hooks, 2 when the loud-failure hook ran
static int call(const ds41a::Fp8Mat& W, int rows, bool ready, int* xf_n, int* gv_n, bool wide = false)
{
    g_xf_calls = g_gv_calls = g_fail_calls = 0;
    uint32_t xf[4];
    ds41dsp::dsp_mm(W, nullptr, 5120, nullptr, 160, xf, ready, nullptr, 512, rows, 1, 0, 0, 0, nullptr, wide);
    *xf_n = g_xf_calls;
    *gv_n = g_gv_calls;
    return g_fail_calls ? 2 : (g_xf_calls == 0 && g_gv_calls == 0);
}

int main()
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0)
    {
        std::printf("a CUDA device is visible: run with CUDA_VISIBLE_DEVICES= (this test must not touch a GPU)\n");
        return 2;
    }
    (void)cudaGetLastError();
    static const uint8_t dummy[16] = {};
    ds41a::Fp8Mat with_gv{dummy, dummy, 512, 5120, dummy}, no_gv{dummy, dummy, 512, 5120, nullptr};
    int xn, gn;

    ds41dsp::g_dsp_fail = &stub_fail;
    ds41a::g_use_gv = true;
    ds41a::g_fn_xfrag = &stub_xfrag;
    ds41a::g_fn_gemv = &stub_gemv;

    // env on
    ds41dsp::g_dsp_gv = 1;
    for (int rows : {1, 5, 6, 8})
    {
        EXPECT(call(with_gv, rows, false, &xn, &gn) == 0 && xn == 1 && gn == 1, "env on, %d rows: xfrag %d gemv %d", rows, xn, gn);
        EXPECT(call(with_gv, rows, true, &xn, &gn) == 0 && xn == 0 && gn == 1, "env on, %d rows, table ready: xfrag %d gemv %d", rows, xn, gn);
    }
    EXPECT(call(with_gv, 9, false, &xn, &gn) == 2 && xn == 0 && gn == 0, "env on, 9 rows -> loud failure: xfrag %d gemv %d", xn, gn);
    EXPECT(call(with_gv, 0, false, &xn, &gn) == 2 && xn == 0 && gn == 0, "env on, 0 rows -> loud failure");
    EXPECT(call(no_gv, 5, false, &xn, &gn) == 2 && xn == 0 && gn == 0, "env on, no gv records -> loud failure");
    // prefill launches (wide): more than 8 rows is the GEMM, 8 rows and fewer keep the GEMV, a matrix that cannot take the GEMV at <= 8 rows still fails loudly
    for (int rows : {9, 32, 128})
    {
        EXPECT(call(with_gv, rows, false, &xn, &gn, true) == 1 && xn == 0 && gn == 0, "env on, wide, %d rows -> GEMM: xfrag %d gemv %d", rows, xn, gn);
        EXPECT(call(no_gv, rows, false, &xn, &gn, true) == 1 && xn == 0 && gn == 0, "env on, wide, no gv records, %d rows -> GEMM", rows);
        EXPECT(call(with_gv, rows, false, &xn, &gn, false) == 2 && xn == 0 && gn == 0, "env on, NOT wide, %d rows -> loud failure", rows);
    }
    for (int rows : {1, 8})
        EXPECT(call(with_gv, rows, false, &xn, &gn, true) == 0 && xn == 1 && gn == 1, "env on, wide, %d rows -> GEMV: xfrag %d gemv %d", rows, xn, gn);
    EXPECT(call(no_gv, 8, false, &xn, &gn, true) == 2 && xn == 0 && gn == 0, "env on, wide, 8 rows, no gv records -> still loud failure");
    EXPECT(call(with_gv, 0, false, &xn, &gn, true) == 2 && xn == 0 && gn == 0, "env on, wide, 0 rows -> loud failure");
    ds41a::g_use_gv = false;
    EXPECT(call(with_gv, 5, false, &xn, &gn) == 2 && xn == 0 && gn == 0, "env on, g_use_gv off -> loud failure");
    ds41a::g_use_gv = true;
    ds41a::g_fn_gemv = nullptr;
    EXPECT(call(with_gv, 5, false, &xn, &gn) == 2 && xn == 0 && gn == 0, "env on, hooks unbound -> loud failure");
    ds41a::g_fn_gemv = &stub_gemv;

    // knob off: every case is the launch_gemm call
    ds41dsp::g_dsp_gv = 0;
    for (int rows : {1, 5, 6, 8, 9})
        EXPECT(call(with_gv, rows, false, &xn, &gn) == 1 && xn == 0 && gn == 0, "env off, %d rows -> GEMM: xfrag %d gemv %d", rows, xn, gn);
    EXPECT(!ds41dsp::dsp_use_gv(with_gv, 5), "env off: dsp_use_gv must be false");


    // GR8_DS_DSP_TC: the attention block's tensor-core entry. Observed through the stub's call count (dsp_mm's TC branch is the only caller) and, for the decline path, through
    // dsp_try_tc's return value (false = nothing launched = dsp_mm's else-branch runs the scalar GEMM; a CUDA launch error cannot be used here: with no
    // device every runtime call already returns 100). The scalar GEMM branch itself is the unchanged final else of the chain.
    {
        ds41dsp::g_dsp_gv = 0;
        uint32_t xf[4];
        auto entry_calls = [&](int rows, bool wide, int groups) {
            g_tc_calls = 0;
            ds41dsp::dsp_mm(with_gv, nullptr, 5120, nullptr, 160, xf, false, nullptr, 512, rows, groups, 0, 0, 0, nullptr, wide);
            return g_tc_calls;
        };
        ds41dsp::g_dsp_tc = 0;
        EXPECT(entry_calls(128, true, 1) == 0, "TC env off, wide 128 rows: entry must not be called");
        EXPECT(!ds41dsp::dsp_try_tc(with_gv, nullptr, 5120, nullptr, 160, nullptr, 512, 128, nullptr), "TC env off: dsp_try_tc must be false");
        ds41dsp::g_dsp_tc = 1;
#ifndef ROUTE_NO_TC_STUB
        g_tc_rc = 0;
        EXPECT(entry_calls(128, true, 1) == 1, "TC on, rc 0, wide 128 rows: entry exactly once");
        EXPECT(g_tc_M == 128 && g_tc_N == 512 && g_tc_K == 5120 && g_tc_lda == 5120 && g_tc_lsa == 160 && g_tc_ldy == 512,
               "TC on: arguments M %d N %d K %d lda %d lsa %d ldy %d", g_tc_M, g_tc_N, g_tc_K, g_tc_lda, g_tc_lsa, g_tc_ldy);
        const int taken0 = ds41dsp::g_dsp_tc_rc[0];
        EXPECT(ds41dsp::dsp_try_tc(with_gv, nullptr, 5120, nullptr, 160, nullptr, 512, 128, nullptr), "rc 0 -> dsp_try_tc true (launched)");
        EXPECT(ds41dsp::g_dsp_tc_rc[0] == taken0 + 1, "rc 0 counted");
        for (int rc : {1, 2})
        {
            g_tc_rc = rc;
            const int before = ds41dsp::g_dsp_tc_rc[rc];
            EXPECT(entry_calls(128, true, 1) == 1, "TC on, rc %d: entry called once", rc);
            EXPECT(!ds41dsp::dsp_try_tc(with_gv, nullptr, 5120, nullptr, 160, nullptr, 512, 128, nullptr), "rc %d -> dsp_try_tc false (scalar GEMM runs)", rc);
            EXPECT(ds41dsp::g_dsp_tc_rc[rc] == before + 2, "rc %d counted twice", rc);
        }
        g_tc_rc = 7;
        EXPECT(!ds41dsp::dsp_try_tc(with_gv, nullptr, 5120, nullptr, 160, nullptr, 512, 128, nullptr), "unknown rc 7 -> false");
        g_tc_rc = -1;
        EXPECT(!ds41dsp::dsp_try_tc(with_gv, nullptr, 5120, nullptr, 160, nullptr, 512, 128, nullptr), "rc -1 -> false");
        g_tc_rc = 0;
        EXPECT(entry_calls(128, false, 1) == 0, "TC on, NOT wide: entry must not be called");
        EXPECT(entry_calls(8, true, 1) == 0, "TC on, wide 8 rows: entry must not be called");
        EXPECT(entry_calls(1, true, 1) == 0, "TC on, wide 1 row: entry must not be called");
        EXPECT(entry_calls(128, true, 2) == 0, "TC on, 2 groups: entry must not be called");
        EXPECT(entry_calls(9, true, 1) == 1, "TC on, wide 9 rows: entry once");
        // GV env on: a wide launch of > 8 rows reaches the entry too (the prefill route); a non-wide one stays the loud failure
        ds41dsp::g_dsp_gv = 1;
        EXPECT(entry_calls(128, true, 1) == 1, "TC on + GV on, wide 128 rows: entry once");
        g_fail_calls = 0;
        EXPECT(entry_calls(128, false, 1) == 0 && g_fail_calls == 1, "TC on + GV on, NOT wide 128 rows: loud failure, entry not called");
        EXPECT(entry_calls(5, true, 1) == 0, "TC on + GV on, wide 5 rows: GEMV, entry not called");
        ds41dsp::g_dsp_gv = 0;
#else
        EXPECT(!ds41dsp::dsp_try_tc(with_gv, nullptr, 5120, nullptr, 160, nullptr, 512, 128, nullptr), "TC on, entry absent (weak, null): dsp_try_tc false");
#endif
        ds41dsp::g_dsp_tc = 0;
    }

    // MainS: the two new tables sit inside bytes(M) and do not overlap the other buffers
    for (int M : {1, 8, 32, 128})
    {
        const size_t n = ds41dsp::MainS::bytes(M);
        uint8_t* base = (uint8_t*)0x100000;
        ds41dsp::MainS S;
        S.carve(base, M);
        const uint8_t* e_tap = (const uint8_t*)S.xf_tap + (size_t)15360 * 16;
        const uint8_t* e_h = (const uint8_t*)S.xf_h + (size_t)5120 * 16;
        EXPECT(e_h <= base + n, "M %d: xf_h ends at +%zu, bytes %zu", M, (size_t)(e_h - base), n);
        EXPECT((const uint8_t*)S.xf_tap >= (const uint8_t*)S.mk + (size_t)M * 512 * 2, "M %d: xf_tap overlaps mk", M);
        EXPECT((const uint8_t*)S.xf_h >= e_tap, "M %d: xf_h overlaps xf_tap", M);
        EXPECT(((uintptr_t)S.xf_tap & 255) == 0 && ((uintptr_t)S.xf_h & 255) == 0, "M %d: xf tables not 256-aligned", M);
    }
    std::printf("%s (%d failures)\n", g_fails ? "ds41_dsp_route_test FAIL" : "ds41_dsp_route_test PASS", g_fails);
    return g_fails ? 1 : 0;
}
