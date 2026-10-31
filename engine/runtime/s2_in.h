// S2 step layout (S2In) and its block copy into shared memory. Host-compilable (no CUDA headers) so a g++ test
// (runtime/s2/tests/ds41_s32_step_test.cpp) checks the layout and the copy that k_s2_cells / k_s2_draft_hid / k_s2_draft_keep run.
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define G6_S2IN_HD __host__ __device__
#else
#define G6_S2IN_HD
#endif

namespace g6rt
{

#if defined(G6_MODEL_DS41) && G6_MODEL_DS41
constexpr int S2N = 32;               // session slots: g6s2::MAX_SESSIONS
#else
constexpr int S2N = 8;
#endif

// per-step session layout. ns sessions in the step; j-th: slot sess[j], rows row0[j]..row0[j]+rows[j]-1, context ctx[j]
// (sessions in ascending slot order, rows contiguous, row0[0] = 0)
struct S2In
{
    int ns, cls;                      // cls: the step's k1 row class (1 / 3 / 5)
    int sess[S2N], row0[S2N], rows[S2N], ctx[S2N];
    int src0[S2N];                    // draft model, first form: the j-th session's first row in the main model's x
};
constexpr int S2IN_WORDS = int(sizeof(S2In) / 4);
static_assert(sizeof(S2In) % 4 == 0 && S2IN_WORDS == 2 + 5 * S2N, "S2In is 2 + 5 x S2N 32-bit words");

// one thread's share of copying the S2In at src into dst, nthreads threads per block: strided, so every word is copied
// for any block size (at S2N = 32 the struct is 162 words, more than a 128-thread block)
G6_S2IN_HD inline void s2in_copy(uint32_t* dst, const uint32_t* src, int t, int nthreads)
{
    for (int i = t; i < S2IN_WORDS; i += nthreads)
#ifdef __CUDA_ARCH__
        dst[i] = __ldcv(src + i);
#else
        dst[i] = src[i];
#endif
}

}   // namespace g6rt
