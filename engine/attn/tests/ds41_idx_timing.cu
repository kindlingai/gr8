// DS41 attention B (indexer) timing harness: per-kernel span of the indexer stages vs the number of compressed keys per query row.
// Synthetic index-K rows (random fp4 codes, E8M0 scales near 1), R query rows all seeing N keys. Spans are cudaEvent averages over
// REP launches of each kernel in isolation, in stream order (the kernels read the previous kernel's real output, so topk sees real logits).
// Needs a GPU that runs sm_121a (a Spark). Build and run:
//   nvcc -std=c++17 -gencode=arch=compute_121a,code=sm_121a -I. -Iengine engine/attn/tests/ds41_idx_timing.cu -o /tmp/ds41_idx_timing
//   /tmp/ds41_idx_timing            # N list 2K..1M, R = 1 and 6
#include "../ds41_idx.cuh"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace ds41i;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static cudaEvent_t e0, e1;
template <class F> static double span_us(F f, int rep = 10)
{
    for (int i = 0; i < 3; i++) f();
    CK(cudaEventRecord(e0));
    for (int i = 0; i < rep; i++) f();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return 1000.0 * ms / rep;
}

int main()
{
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    cudaStream_t st = 0;
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("device %s sm_%d%d, %d SMs\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount);
    const int Ns[] = {2048, 8192, 32768, 131072, 262144, 1048576};
    const int Rs[] = {1, 6};
    const int NMAX = 1048576, RMAX = 8;
    // synthetic ica rows (flat, rpp = NMAX): codes plane then scale plane
    std::mt19937 rng(7);
    std::vector<uint8_t> flat((size_t)NMAX * 68);
    for (size_t i = 0; i < (size_t)NMAX * 64; i++) flat[i] = (uint8_t)rng();
    for (size_t i = 0; i < (size_t)NMAX * 4; i++) flat[(size_t)NMAX * 64 + i] = (uint8_t)(124 + rng() % 6);
    uint8_t* ica;
    CK(cudaMalloc(&ica, flat.size()));
    CK(cudaMemcpy(ica, flat.data(), flat.size(), cudaMemcpyHostToDevice));
    int zero = 0;
    int* page0;
    CK(cudaMalloc(&page0, 4));
    CK(cudaMemcpy(page0, &zero, 4, cudaMemcpyHostToDevice));
    std::vector<uint16_t> hq((size_t)RMAX * 4096), hw(RMAX * 32);
    auto bfr = [&](float f) { uint32_t x; memcpy(&x, &f, 4); return (uint16_t)(x >> 16); };
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hq) v = bfr(nd(rng));
    for (auto& v : hw) v = bfr(nd(rng) * 0.02f);
    bf16 *dq, *dw;
    CK(cudaMalloc(&dq, hq.size() * 2));
    CK(cudaMalloc(&dw, hw.size() * 2));
    CK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    void* scratch;
    CK(cudaMalloc(&scratch, Scratch::bytes(RMAX, NMAX)));
    int *dpos, *klen, *out, *out2;
    CK(cudaMalloc(&dpos, RMAX * 4));
    CK(cudaMalloc(&klen, RMAX * 4));
    CK(cudaMalloc(&out, RMAX * 512 * 4));
    CK(cudaMalloc(&out2, RMAX * 512 * 4));

    printf("%-9s %-2s | %-9s %-9s %-9s %-9s | %-9s %-9s | %-9s %-9s %-9s %-9s | us\n", "keys N", "R", "idx_q", "logits", "topk512", "DENSE", "blk_score",
           "topk2048", "gather", "cmp_lens", "topk512c", "CONSUMER");
    for (int N : Ns)
        for (int R : Rs)
        {
            std::vector<int> hp(R, N - 1), hk(R, N);
            CK(cudaMemcpy(dpos, hp.data(), R * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(klen, hk.data(), R * 4, cudaMemcpyHostToDevice));
            const Scratch sc = Scratch::carve(scratch, RMAX, NMAX);
            IcaMap km = ica_map(ica, page0, NMAX, (size_t)NMAX * 68);
            IdxQP q{dq, dw, rope_inv_ds41(), 0, sc.q_pack, sc.q_exp, sc.w, dpos};
            LogitsP l{};
            l.q_pack = sc.q_pack; l.q_exp = sc.q_exp; l.w = sc.w; l.k = km; l.klen = klen; l.p = sc.p; l.ldp = sc.ldp; l.ncols = N; l.R = R;
            const int nb = (int)sc.ldsc;
            const double t_q = span_us([&] { launch_index_q(q, R, st); });
            const double t_l = span_us([&] { launch_logits_dense(l, st); });
            const double t_t = span_us([&] { launch_topk<TOPK>(sc.p, sc.ldp, klen, (int)sc.ldp, out, 512, R, st); });
            const double t_b = span_us([&] { launch_block_scores(sc.p, sc.ldp, klen, nb, sc.sc, sc.ldsc, sc.nvalid, R, st); });
            const double t_c = span_us([&] { launch_topk<CAND_BLOCKS>(sc.sc, sc.ldsc, sc.nvalid, nb, sc.cand, CAND_BLOCKS, R, st); });
            LogitsP g = l;
            g.cand = sc.cand; g.ldc = CAND_BLOCKS; g.p = sc.cp; g.ldp = CAND_COLS; g.ncols = CAND_COLS;
            const double t_g = span_us([&] { launch_logits_gather(g, st); });
            const double t_n = span_us([&] { launch_compact_lens(sc.cand, CAND_BLOCKS, klen, sc.clen, R, st); });
            const double t_k = span_us([&] { launch_topk<TOPK>(sc.cp, CAND_COLS, sc.clen, CAND_COLS, sc.ci, TOPK, R, st); });
            const double t_r = span_us([&] { launch_restore(sc.ci, TOPK, sc.cand, CAND_BLOCKS, out2, 512, R, st); });
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            printf("%-9d %-2d | %-9.1f %-9.1f %-9.1f %-9.1f | %-9.1f %-9.1f | %-9.1f %-9.1f %-9.1f %-9.1f | us (restore %.1f)\n", N, R, t_q, t_l, t_t,
                   t_q + t_l + t_t, t_b, t_c, t_g, t_n, t_k, t_q + t_g + t_n + t_k + t_r, t_r);
        }
    return 0;
}
