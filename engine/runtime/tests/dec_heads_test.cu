// The head-batched decode attention kernels against the ones they replace: head-batched scores and context against the per-head
// kernels, absorb_m / value_project_m over all rows (and merging the context parts) against the per-row kernels. Random
// KV pages, index lists with dead slots and dead tails, slots past the row's position, repeated keys, random q / rope /
// probabilities (some exactly 0). Every output byte must match. One GPU, no pack files.
//   scripts/c1_tests.sh <bin_dir> (TEST=dec_heads_test; --bench prints kernel times)
// Exit code: 0 all PASS, 1 any FAIL, 2 CUDA error.
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "../../attn/attn.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

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

static int fails = 0, passes = 0;
static void check(const char* what, const void* x, const void* y, size_t n, int H, int rows, int ctx)
{
    const bool ok = !memcmp(x, y, n);
    ok ? ++passes : ++fails;
    printf("%s %s heads %d rows %d context %d\n", ok ? "PASS" : "FAIL", what, H, rows, ctx);
}

template <class F> static float time_ms(F f, int it = 200)
{
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f();
    CK(cudaEventRecord(a));
    for (int i = 0; i < it; ++i) f();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    return ms * 1000 / it;      // us per call
}

int main(int argc, char** argv)
{
    const bool bench = argc > 1 && !strcmp(argv[1], "--bench");
    CK((cudaError_t)g6_attn_rope_init(4096, nullptr));      // as a decode boot: absorb reads the RoPE table
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    const int TOP = ga::TOP, PART = ga::PART, KVB = ga::KV_BYTES, PS = 64;
    for (int H : {16, 11, 10})
        for (int rows : {1, 3, 8, 24})
            for (int ctx : {37, 500, 2048, 3000})
            {
                const int pages = (ctx + PS - 1) / PS;
                std::vector<uint8_t> cache((size_t)pages * PS * KVB);
                for (size_t p = 0; p < (size_t)pages * PS; ++p)
                {
                    uint8_t* v = cache.data() + p * KVB;
                    for (int d = 0; d < 512; ++d) { uint8_t b; do b = rng() & 0xff; while ((b & 0x7f) == 0x7f); v[d] = b; }
                    for (int g = 0; g < 4; ++g) { float s = 0.01f + (rng() % 1000) * 1e-4f; memcpy(v + 512 + 4 * g, &s, 4); }
                    for (int d = 0; d < 64; ++d) { __nv_bfloat16 b = __float2bfloat16(uf(rng)); memcpy(v + 528 + 2 * d, &b, 2); }
                }
                std::vector<int> table(pages), pos(rows), idx((size_t)rows * TOP);
                for (int i = 0; i < pages; ++i) table[i] = i;
                for (int r = 0; r < rows; ++r)
                {
                    pos[r] = ctx - rows + r;
                    for (int j = 0; j < TOP; ++j)
                    {
                        const int k = rng() % 10;
                        idx[(size_t)r * TOP + j] = k == 0 ? -1 : k == 1 ? ctx - 1 : (j < ctx ? j : ctx < 1000 ? -1 : (int)(rng() % ctx));   // short contexts: dead tail, as at decode
                    }
                }
                std::vector<float> q((size_t)rows * H * 512), rope((size_t)rows * H * 64), prob((size_t)rows * H * TOP);
                for (float& x : q) x = uf(rng);
                for (float& x : rope) x = uf(rng);
                for (float& x : prob) x = rng() % 3 == 0 ? 0.f : (uf(rng) + 1.f) * 1e-3f;
                G6Attn a{};
                a.rows = rows; a.heads = H; a.context = ctx; a.page_size = PS; a.page_count = pages; a.table_size = pages;
                a.query_tile = 32;
                a.positions = up(pos); a.block_table = up(table); a.mla_cache = up(cache); a.indices = up(idx);
                unsigned* err; CK(cudaMalloc(&err, 4)); CK(cudaMemset(err, 0, 4)); a.error = err;
                float *dq = up(q), *dr = up(rope), *dp = up(prob);
                const size_t ns = (size_t)rows * H * TOP, np = (size_t)rows * H * PART * 512;
                float *s0, *s1, *p0, *p1, *p2;
                CK(cudaMalloc(&s0, ns * 4)); CK(cudaMalloc(&s1, ns * 4));
                CK(cudaMalloc(&p0, np * 4)); CK(cudaMalloc(&p1, np * 4)); CK(cudaMalloc(&p2, np * 4));
                CK(cudaMemset(s1, 0x55, ns * 4)); CK(cudaMemset(p1, 0x55, np * 4)); CK(cudaMemset(p2, 0x55, np * 4));
                ga::small_decode::scores_direct<<<dim3(TOP / 8, H, rows), 256>>>(a, 0, dq, dr, s0);
                ga::small_decode::scores_heads<16><<<dim3(TOP / 8, (H + 15) / 16, rows), 256>>>(a, 0, dq, dr, s1);
                ga::small_decode::context_parts_cpb<<<dim3(PART * 2, H, rows), 128>>>(a, 0, dp, p0);
                ga::small_decode::context_parts_heads<16><<<dim3(PART * 2, (H + 15) / 16, rows), 128>>>(a, 0, dp, p1);
                ga::small_decode::context_parts_heads<4><<<dim3(PART * 2, (H + 3) / 4, rows), 128>>>(a, 0, dp, p2);
                // absorb / value_project: the per-row kernels against absorb_m / value_project_m over all rows at once
                std::vector<__half> qb((size_t)rows * H * 256);
                for (auto& x : qb) x = __float2half(uf(rng));
                std::vector<__nv_bfloat16> uk((size_t)H * 512 * 192), uv((size_t)H * 256 * 512);
                for (auto& x : uk) x = __float2bfloat16(uf(rng) * 0.1f);
                for (auto& x : uv) x = __float2bfloat16(uf(rng) * 0.1f);
                a.q_b = up(qb); a.uk = up(uk); a.uv = up(uv);
                const size_t nq = (size_t)rows * H * 512, nr = (size_t)rows * H * 64, no = (size_t)rows * H * 256;
                float *q0, *q1, *r0, *r1;
                __half *o0, *o1;
                CK(cudaMalloc(&q0, nq * 4)); CK(cudaMalloc(&q1, nq * 4)); CK(cudaMalloc(&r0, nr * 4)); CK(cudaMalloc(&r1, nr * 4));
                CK(cudaMalloc(&o0, no * 2)); CK(cudaMalloc(&o1, no * 2));
                ga::absorb<<<dim3(64, H, rows), 256>>>(a, 0, q0, r0);
                ga::absorb_m<<<dim3(64, H), 256>>>(a, 0, rows, q1, r1);
                a.output = o0; ga::value_project<<<dim3(32, H, rows), 256>>>(a, 0, dq);
                a.output = o1; ga::value_project_m<<<dim3(32, H), 256>>>(a, 0, rows, dq);
                CK(cudaDeviceSynchronize());
                check("absorb_m all rows", down(q0, nq).data(), down(q1, nq).data(), nq * 4, H, rows, ctx);
                check("absorb_m rope", down(r0, nr).data(), down(r1, nr).data(), nr * 4, H, rows, ctx);
                check("value_project_m all rows", down(o0, no).data(), down(o1, no).data(), no * 2, H, rows, ctx);
                if (bench && H == 16)
                {
                    const float t0 = time_ms([&] { ga::small_decode::scores_direct<<<dim3(TOP / 8, H, rows), 256>>>(a, 0, dq, dr, s0); });
                    const float t1 = time_ms([&] { ga::small_decode::scores_heads<16><<<dim3(TOP / 8, 1, rows), 256>>>(a, 0, dq, dr, s1); });
                    const float c0 = time_ms([&] { ga::small_decode::context_parts_cpb<<<dim3(PART * 2, H, rows), 128>>>(a, 0, dp, p0); });
                    const float c4 = time_ms([&] { ga::small_decode::context_parts_heads<4><<<dim3(PART * 2, H / 4, rows), 128>>>(a, 0, dp, p1); });
                    const float c8 = time_ms([&] { ga::small_decode::context_parts_heads<8><<<dim3(PART * 2, H / 8, rows), 128>>>(a, 0, dp, p1); });
                    const float c16 = time_ms([&] { ga::small_decode::context_parts_heads<16><<<dim3(PART * 2, 1, rows), 128>>>(a, 0, dp, p1); });
                    const float ab = time_ms([&] { ga::absorb_m<<<dim3(64, H), 256>>>(a, 0, rows, q1, r1); });
                    const float vp = time_ms([&] { ga::value_project_m<<<dim3(32, H), 256>>>(a, 0, rows, dq); });
                    const float tg = time_ms([&] { ga::mla_scores<<<dim3(TOP / 8, H, rows), 256>>>(a, 0, dq, dr, s0); });
                    const float cg = time_ms([&] { ga::context_parts<<<dim3(PART, H, rows), 256>>>(a, 0, dp, p0); });
                    printf("BENCH rows %d ctx %d: generic mla_scores %.1f us, generic context_parts %.1f us\n", rows, ctx, tg, cg);
                    printf("BENCH rows %d ctx %d: scores %.1f -> %.1f us; context %.1f -> HG4 %.1f HG8 %.1f HG16 %.1f us; absorb_m %.1f vproj_m %.1f us\n",
                           rows, ctx, t0, t1, c0, c4, c8, c16, ab, vp);
                }
                for (void* p : {(void*)a.q_b, (void*)a.uk, (void*)a.uv, (void*)q0, (void*)q1, (void*)r0, (void*)r1, (void*)o0, (void*)o1}) CK(cudaFree(p));
                const auto S0 = down(s0, ns), S1 = down(s1, ns);
                const auto P0 = down(p0, np), P1 = down(p1, np), P2 = down(p2, np);
                check("scores_heads<16>", S0.data(), S1.data(), ns * 4, H, rows, ctx);
                check("context_parts_heads<16>", P0.data(), P1.data(), np * 4, H, rows, ctx);
                check("context_parts_heads<4>", P0.data(), P2.data(), np * 4, H, rows, ctx);
                {   // the generic kernels (fresh-index verify rows, contexts past 2048) against the same outputs
                    float *g0, *g1;
                    CK(cudaMalloc(&g0, ns * 4)); CK(cudaMalloc(&g1, np * 4));
                    ga::mla_scores<<<dim3(TOP / 8, H, rows), 256>>>(a, 0, dq, dr, g0);
                    ga::context_parts<<<dim3(PART, H, rows), 256>>>(a, 0, dp, g1);
                    CK(cudaDeviceSynchronize());
                    check("scores_heads<16> vs mla_scores", down(g0, ns).data(), down(s1, ns).data(), ns * 4, H, rows, ctx);
                    check("context_parts_heads<16> vs context_parts", down(g1, np).data(), down(p1, np).data(), np * 4, H, rows, ctx);
                    CK(cudaFree(g0)); CK(cudaFree(g1));
                }
                for (void* p : {(void*)a.positions, (void*)a.block_table, a.mla_cache, (void*)a.indices, (void*)err, (void*)dq, (void*)dr,
                                (void*)dp, (void*)s0, (void*)s1, (void*)p0, (void*)p1, (void*)p2})
                    CK(cudaFree(p));
            }
    printf("dec_heads_test: %d PASS, %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
