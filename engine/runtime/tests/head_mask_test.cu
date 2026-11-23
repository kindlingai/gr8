// Regression test for grammar-masked vocab-split head kernels (glue.cuh, head2.cuh):
// simulates NR ranks (default TP=6) on one GPU with a loopback fabric where 5/6 ranks have
// 100% -INFINITY logits and 1 rank has 1 (or a few) unmasked tokens, covering:
//   - Issue #576: SPEC_ACCEPT=rejection (k_head_local_rs / k_head_merge_rs / k_head_merge_tp_rs)
//     with d == only allowed token, d != allowed token (d masked), negative sole logit,
//     sole nucleus token (other unmasked tokens outside top_p), sparse multi-rank > TP_RANGE,
//     T=0 greedy, T=0.7, top_p=0.9 (nc=64), top_p=1.0 (nc=0), topk > 0 (TkRes), target lpt_out,
//     coupled kernels (k_head_local), relaxed kernels (k_head_local_rx + k_head_relax),
//     and head2 (k_screen_top, k_head_local_sl, k_head_local_sl_map) on all--inf / sparse rows.
//
// Build (example):
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_NR=6 -DDEC_S2 \
//     -I engine/kernels -o head_mask_test engine/runtime/tests/head_mask_test.cu
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../../kernels/head2.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

template <typename T>
static T* dalloc(size_t n)
{
    T* p = nullptr;
    CK(cudaMalloc((void**)&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

constexpr int NR = g6nr::NR;
constexpr int M = 8;
constexpr int VN = 2048;   // > nc (64) so head_local_topp histogram path runs identically to production
constexpr int V = NR * VN;

struct Loop
{
    const char* send[NR];
    char* recv;
    const u64* stepp;
    int ncoll;
    int n16;
    size_t stride;
};

__global__ void k_loop(Loop a)
{
    const u64 step = *a.stepp;
    const size_t slot = (step & 1) * a.ncoll;
    const uint4* s = reinterpret_cast<const uint4*>(a.send[blockIdx.x] + slot * a.stride);
    uint4* d = reinterpret_cast<uint4*>(a.recv + (slot * NR + blockIdx.x) * a.stride);
    for (int i = threadIdx.x; i < a.n16; i += 128) d[i] = s[i];
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        *reinterpret_cast<u64*>(d + a.n16) = g6::coll_seq(step, 0, a.ncoll);
    }
}

__global__ void k_set(u64* p, u64 v) { *p = v; }

static double ref_lse(const float* row, int n)
{
    double mx = -INFINITY;
    for (int i = 0; i < n; ++i)
        if (row[i] > mx) mx = row[i];
    if (!std::isfinite(mx)) return -INFINITY;
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        if (row[i] != -INFINITY) s += std::exp((double)row[i] - mx);
    return mx + std::log(s);
}

static bool test_head2_masked()
{
    // Test head2::k_screen_top, k_head_local_sl, k_head_local_sl_map with row 0 = all -INFINITY,
    // row 1 = single finite logit, row 2 = normal logits.
    constexpr int HM = 3;
    constexpr int HVN = 2048;
    std::vector<float> hy((size_t)HM * HVN, -INFINITY);
    hy[1 * HVN + 1024 + 77] = -12.5f;   // >= 1024 so thread 77 sees -INFINITY first!
    for (int i = 0; i < HVN; ++i) hy[2 * HVN + i] = -0.01f * (float)i;

    float* dy = dalloc<float>((size_t)HM * HVN);
    CK(cudaMemcpy(dy, hy.data(), hy.size() * sizeof(float), cudaMemcpyHostToDevice));
    int* dcand = dalloc<int>((size_t)HM * head2::SL);
    float* dcl = dalloc<float>((size_t)HM * head2::SL);
    std::vector<float> hcl((size_t)HM * head2::SL, -INFINITY);
    hcl[1 * head2::SL + 0] = -12.5f;
    for (int i = 0; i < head2::SL; ++i) hcl[2 * head2::SL + i] = -0.01f * (float)i;
    CK(cudaMemcpy(dcl, hcl.data(), hcl.size() * sizeof(float), cudaMemcpyHostToDevice));

    head2::k_screen_top<<<HM, 1024>>>(dy, HVN, HVN, dcand);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    std::vector<int> hcand((size_t)HM * head2::SL);
    CK(cudaMemcpy(hcand.data(), dcand, hcand.size() * sizeof(int), cudaMemcpyDeviceToHost));
    bool fail = false;
    for (int m = 0; m < HM; ++m)
        for (int j = 0; j < head2::SL; ++j)
            if (hcand[(size_t)m * head2::SL + j] < 0 || hcand[(size_t)m * head2::SL + j] >= HVN)
            {
                printf("FAIL head2 k_screen_top bad cand row %d slot %d = %d\n", m, j, hcand[(size_t)m * head2::SL + j]);
                fail = true;
            }

    const size_t pay = glue::head_bytes(HM, 0), stride = (pay + 8 + 255) / 256 * 256;
    char* send = dalloc<char>(2 * stride);
    u64* stepp = dalloc<u64>(1);
    unsigned* ticket = dalloc<unsigned>(1);
    unsigned long long* bell = dalloc<unsigned long long>(1);
    unsigned* err = dalloc<unsigned>(1);
    glue::SampleRow hsr[HM];
    for (int m = 0; m < HM; ++m) hsr[m] = glue::SampleRow{1.f / 0.7f, 123u, 0u, 10 + m, 0, 0.f, 0, 0};
    glue::SampleRow* dsr = dalloc<glue::SampleRow>(HM);
    CK(cudaMemcpy(dsr, hsr, sizeof(hsr), cudaMemcpyHostToDevice));
    int* rmap = dalloc<int>(HM);
    const int hrmap[HM] = {0, 1, 2};
    CK(cudaMemcpy(rmap, hrmap, sizeof(hrmap), cudaMemcpyHostToDevice));

    g6::FabricDev f{};
    f.send = send; f.recv = send; f.doorbell = bell; f.error = err;
    f.rank = 0; f.nr = 1; f.ncoll = 1; f.stride = stride;
    glue::HeadArgs a{};
    a.f = f; a.stepp = stepp; a.c = 0; a.M = HM; a.logits = dy; a.ldl = HVN; a.vlo = 0; a.vn = HVN;
    a.rows = dsr; a.ticket = ticket;

    for (int variant = 0; variant < 2; ++variant)
    {
        k_set<<<1, 1>>>(stepp, (u64)(variant + 1));
        if (variant == 0) head2::k_head_local_sl<<<HM, 1024>>>(a, dcand, dcl);
        else head2::k_head_local_sl_map<<<HM, 1024>>>(a, dcand, dcl, rmap);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());

        glue::HeadRec rec[HM];
        const size_t slot = ((variant + 1) & 1) * stride;
        CK(cudaMemcpy(rec, send + slot, sizeof(rec), cudaMemcpyDeviceToHost));
        if (rec[0].lmax != -INFINITY || rec[0].lsum != 0.f || std::isnan(rec[0].lsum))
        {
            printf("FAIL head2 %s row 0 (all -inf): lmax=%g lsum=%g (want -inf, 0)\n",
                   variant ? "sl_map" : "sl", rec[0].lmax, rec[0].lsum);
            fail = true;
        }
        if (std::fabs(rec[1].lmax - (-12.5f)) > 1e-5f || std::fabs(rec[1].lsum - 1.f) > 1e-5f || std::isnan(rec[1].lsum))
        {
            printf("FAIL head2 %s row 1 (1 finite): lmax=%g lsum=%g (want -12.5, 1)\n",
                   variant ? "sl_map" : "sl", rec[1].lmax, rec[1].lsum);
            fail = true;
        }
        if (!std::isfinite(rec[2].lmax) || !std::isfinite(rec[2].lsum) || rec[2].lsum <= 0.f)
        {
            printf("FAIL head2 %s row 2 (normal): lmax=%g lsum=%g\n",
                   variant ? "sl_map" : "sl", rec[2].lmax, rec[2].lsum);
            fail = true;
        }
    }

    cudaFree(dy); cudaFree(dcand); cudaFree(dcl); cudaFree(send); cudaFree(stepp);
    cudaFree(ticket); cudaFree(bell); cudaFree(err); cudaFree(dsr); cudaFree(rmap);
    printf("head2 masked checks: %s\n", fail ? "FAIL" : "PASS");
    return !fail;
}

int main()
{
    static_assert(NR >= 4, "head_mask_test expects NR >= 4 (TP=4 or TP=6)");
    constexpr int SEEDS = 64;
    bool fail = false;

    if (!test_head2_masked()) fail = true;

    // Construct 8 grammar-masked rows across NR ranks:
    std::vector<float> lg((size_t)M * V, -INFINITY);
    const int u0 = 0 * VN + 42;                 // row 0: sole allowed on rank 0, d == u0
    const int u1 = (NR / 2) * VN + 13;          // row 1: sole allowed on rank NR/2, very negative logit, d == u1
    const int u2 = (NR - 1) * VN + 99;          // row 2: sole allowed on last rank, d masked on rank 1
    const int u3a = std::min(2, NR - 1) * VN + 7;
    const int u3b = std::min(2, NR - 1) * VN + 1800; // row 3: d == u3a sole nucleus token; u3b outside top_p
    const int u4a = 1 * VN + 5;
    const int u4b = 1 * VN + 6;
    const int u4c = (NR - 2) * VN + 777;        // row 4: sparse across 2 ranks, logit spread > TP_RANGE (32)
    const int u5a = 0 * VN + 333;
    const int u5b = (NR - 1) * VN + 444;        // row 5: 2 ranks unmasked, d == u5b (second best)
    const int u6 = 0 * VN + 1024;               // row 6: sole allowed on rank 0, nodraft=1
    const int u7 = (NR - 1) * VN + 512;         // row 7: sole allowed on last rank, no draft (last row)

    lg[0 * V + u0] = 6.25f;
    lg[1 * V + u1] = -85.0f;
    lg[2 * V + u2] = 3.5f;
    lg[3 * V + u3a] = 10.0f;
    lg[3 * V + u3b] = -30.0f;
    lg[4 * V + u4a] = 5.0f;
    lg[4 * V + u4b] = 4.5f;
    lg[4 * V + u4c] = -40.0f;
    lg[5 * V + u5a] = 4.0f;
    lg[5 * V + u5b] = 3.2f;
    lg[6 * V + u6] = -12.0f;
    lg[7 * V + u7] = 1.75f;

    // Draft tokens: row m's draft is tok[m + 1] (when !nodraft and m + 1 < M)
    int tok[M] = {
        7,
        u0,             // draft for row 0 == u0 (ONLY unmasked token -> Issue #576 exact trigger!)
        u1,             // draft for row 1 == u1 (ONLY unmasked token, logit -85)
        1 * VN + 55,    // draft for row 2 -> MASKED (-INFINITY) on rank 1!
        u3a,            // draft for row 3 == u3a (ONLY token inside top_p=0.9 nucleus)
        u4a,            // draft for row 4 == u4a
        u5b,            // draft for row 5 == u5b
        u6,             // unused by row 6 (nodraft=1)
    };
    // Targets for lpt_out: mix of allowed and masked targets
    int tgt[M] = {u0, 0, u2, u3b, u4c, u5a, 1 * VN + 1, u7};

    double lse_ref[M];
    for (int m = 0; m < M; ++m) lse_ref[m] = ref_lse(&lg[(size_t)m * V], V);

    int* tok_d = dalloc<int>(M);
    int* tgt_d = dalloc<int>(M);
    CK(cudaMemcpy(tok_d, tok, sizeof(tok), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tgt_d, tgt, sizeof(tgt), cudaMemcpyHostToDevice));

    const int ncs[2] = {64, 0};
    const float temps[2] = {0.7f, 0.0f};
    // modes: 0 = coupled (k_head_local), 1 = rejection (k_head_local_rs), 2 = relaxed (k_head_local_rx + k_head_relax)
    for (int nc : ncs)
    {
        for (float T : temps)
        {
            const float inv_temp = T > 0.f ? 1.f / T : 0.f;
            const float tp = nc > 0 ? 0.9f : 0.f;
            const size_t pay = glue::head_bytes_tk(M, nc, true);
            const size_t stride = (pay + 8 + 255) / 256 * 256;
            char* recv = dalloc<char>(2 * (size_t)NR * stride);
            u64* stepp = dalloc<u64>(1);
            unsigned* err = dalloc<unsigned>(1);
            unsigned long long* bell = dalloc<unsigned long long>(1);
            unsigned* miss = dalloc<unsigned>(1);

            std::vector<glue::SampleRow> rows((size_t)SEEDS * M);
            for (int s = 0; s < SEEDS; ++s)
                for (int m = 0; m < M; ++m)
                    rows[(size_t)s * M + m] = glue::SampleRow{
                        inv_temp, (uint32_t)(s * 2654435761u + 1u), (uint32_t)s, 100 + m, 0, tp, 1, m == 6 ? 1 : 0};
            glue::SampleRow* rows_d = dalloc<glue::SampleRow>(rows.size());
            CK(cudaMemcpy(rows_d, rows.data(), rows.size() * sizeof(glue::SampleRow), cudaMemcpyHostToDevice));

            Loop loop{};
            loop.recv = recv; loop.stepp = stepp; loop.ncoll = 1;
            loop.n16 = (int)(pay / 16); loop.stride = stride;

            std::vector<glue::HeadArgs> A(NR);
            std::vector<float*> d_lg(NR);
            std::vector<char*> d_send(NR);
            std::vector<unsigned*> d_tkt(NR);
            std::vector<int*> out(NR);
            std::vector<float*> lpo(NR), lpto(NR);
            std::vector<glue::TkRes*> tko(NR);

            for (int r = 0; r < NR; ++r)
            {
                d_lg[r] = dalloc<float>((size_t)M * VN);
                std::vector<float> sl((size_t)M * VN);
                for (int m = 0; m < M; ++m)
                    memcpy(&sl[(size_t)m * VN], &lg[(size_t)m * V + r * VN], VN * sizeof(float));
                CK(cudaMemcpy(d_lg[r], sl.data(), sl.size() * sizeof(float), cudaMemcpyHostToDevice));
                d_send[r] = dalloc<char>(2 * stride);
                d_tkt[r] = dalloc<unsigned>(1);
                g6::FabricDev f{};
                f.send = d_send[r]; f.recv = recv; f.doorbell = bell; f.error = err;
                f.rank = r; f.nr = NR; f.ncoll = 1; f.stride = stride;
                loop.send[r] = d_send[r];
                glue::HeadArgs& a = A[r];
                a.f = f; a.stepp = stepp; a.c = 0; a.M = M; a.logits = d_lg[r]; a.ldl = VN;
                a.vlo = r * VN; a.vn = VN; a.ticket = d_tkt[r]; a.tgt = tgt_d;
                a.nc = nc; a.topp_miss = miss;
                out[r] = dalloc<int>((size_t)SEEDS * M);
                lpo[r] = dalloc<float>((size_t)SEEDS * M);
                lpto[r] = dalloc<float>((size_t)SEEDS * M);
                tko[r] = dalloc<glue::TkRes>(1);
            }

            const relax::Rule rx_rule{1, -1.0f, nc > 0 ? 0.9f : 0.f};
            u64 step = 0;
            for (int mode = 0; mode < 3; ++mode)
            {
                if (mode == 2 && T > 0.f) continue; // relaxed accept applies to greedy rows
                const char* mname = mode == 0 ? "coupled" : (mode == 1 ? "rejection" : "relaxed");
                for (int s = 0; s < SEEDS; ++s)
                {
                    k_set<<<1, 1>>>(stepp, ++step);
                    for (int r = 0; r < NR; ++r)
                    {
                        glue::HeadArgs a = A[r];
                        a.rows = rows_d + (size_t)s * M;
                        a.tok_out = out[r] + (size_t)s * M;
                        a.lp_out = lpo[r] + (size_t)s * M;
                        a.lpt_out = lpto[r] + (size_t)s * M;
                        a.tk = tko[r];
                        if (mode == 1) glue::k_head_local_rs<<<M, 1024>>>(a, tok_d);
                        else if (mode == 2) glue::k_head_local_rx<<<M, 1024>>>(a, tok_d, rx_rule);
                        else glue::k_head_local<<<M, 1024>>>(a);
                    }
                    k_loop<<<NR, 128>>>(loop);
                    for (int r = 0; r < NR; ++r)
                    {
                        glue::HeadArgs a = A[r];
                        a.rows = rows_d + (size_t)s * M;
                        a.tok_out = out[r] + (size_t)s * M;
                        a.lp_out = lpo[r] + (size_t)s * M;
                        a.lpt_out = lpto[r] + (size_t)s * M;
                        a.tk = tko[r];
                        if (nc > 0 && mode == 1) glue::k_head_merge_tp_rs<<<M, 1024>>>(a, tok_d);
                        else if (nc > 0) glue::k_head_merge_tp<<<M, 1024>>>(a);
                        else if (mode == 1) glue::k_head_merge_rs<<<1, 32>>>(a, tok_d);
                        else glue::k_head_merge<<<1, 32>>>(a);
                        if (mode == 2) glue::k_head_relax<<<M, 256>>>(a, tok_d, rx_rule);
                    }
                }
                CK(cudaDeviceSynchronize());
                CK(cudaGetLastError());

                unsigned e = 0;
                CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
                if (e) { printf("FAIL %s nc=%d T=%g fabric err %08x\n", mname, nc, T, e); fail = true; continue; }

                std::vector<std::vector<int>> ho(NR, std::vector<int>((size_t)SEEDS * M));
                std::vector<std::vector<float>> hl(NR, std::vector<float>((size_t)SEEDS * M));
                std::vector<std::vector<float>> hlt(NR, std::vector<float>((size_t)SEEDS * M));
                for (int r = 0; r < NR; ++r)
                {
                    CK(cudaMemcpy(ho[r].data(), out[r], ho[r].size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hl[r].data(), lpo[r], hl[r].size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(hlt[r].data(), lpto[r], hlt[r].size() * 4, cudaMemcpyDeviceToHost));
                }
                glue::TkRes tk0;
                CK(cudaMemcpy(&tk0, tko[0], sizeof(tk0), cudaMemcpyDeviceToHost));

                int bad = 0;
                for (int r = 1; r < NR; ++r)
                    if (ho[r] != ho[0] || memcmp(hl[r].data(), hl[0].data(), hl[0].size() * 4) != 0)
                    {
                        if (bad++ < 3) printf("FAIL %s nc=%d T=%g rank %d mismatch vs rank 0\n", mname, nc, T, r);
                        fail = true;
                    }

                for (int s = 0; s < SEEDS; ++s)
                {
                    for (int m = 0; m < M; ++m)
                    {
                        const int tid = ho[0][(size_t)s * M + m];
                        const float lp = hl[0][(size_t)s * M + m];
                        const float lpt = hlt[0][(size_t)s * M + m];
                        if (tid < 0 || tid >= V)
                        {
                            if (bad++ < 8)
                                printf("FAIL %s nc=%d T=%g seed=%d row=%d OUT_OF_RANGE tok=%d (0x%x) lp=%g\n",
                                       mname, nc, T, s, m, tid, (unsigned)tid, lp);
                            fail = true;
                            continue;
                        }
                        const float chosen_lg = lg[(size_t)m * V + tid];
                        if (chosen_lg == -INFINITY)
                        {
                            if (bad++ < 8)
                                printf("FAIL %s nc=%d T=%g seed=%d row=%d MASKED_TOKEN tok=%d\n", mname, nc, T, s, m, tid);
                            fail = true;
                            continue;
                        }
                        if (m == 0 && tid != u0) { if (bad++ < 8) printf("FAIL %s row 0 want %d got %d\n", mname, u0, tid); fail = true; }
                        if (m == 1 && tid != u1) { if (bad++ < 8) printf("FAIL %s row 1 want %d got %d\n", mname, u1, tid); fail = true; }
                        if (m == 2 && tid != u2) { if (bad++ < 8) printf("FAIL %s row 2 want %d got %d\n", mname, u2, tid); fail = true; }
                        if (m == 3 && tid != u3a)
                        {
                            if (bad++ < 8) printf("FAIL %s nc=%d T=%g seed=%d row 3 leaked out-of-nucleus tok=%d (want %d)\n",
                                                  mname, nc, T, s, tid, u3a);
                            fail = true;
                        }
                        if (m == 6 && tid != u6) { if (bad++ < 8) printf("FAIL %s row 6 want %d got %d\n", mname, u6, tid); fail = true; }
                        if (m == 7 && tid != u7) { if (bad++ < 8) printf("FAIL %s row 7 want %d got %d\n", mname, u7, tid); fail = true; }
                        const double want_lp = (double)chosen_lg - lse_ref[m];
                        if (!std::isfinite(lp) || std::fabs((double)lp - want_lp) > 1e-4)
                        {
                            if (bad++ < 8)
                                printf("FAIL %s nc=%d T=%g seed=%d row=%d lp=%g want=%g\n", mname, nc, T, s, m, lp, want_lp);
                            fail = true;
                        }
                        // Check lpt_out: note in RS/RX rows with a draft, pad_[0] carries d's logit instead of tgt[m]
                        const bool row_has_draft = (mode == 1 && T > 0.f && m < M - 1 && m != 6) ||
                                                   (mode == 2 && T == 0.f && m < M - 1 && m != 6);
                        const int ref_tid = row_has_draft ? tok[m + 1] : tgt[m];
                        const float ref_tlg = lg[(size_t)m * V + ref_tid];
                        if (ref_tlg == -INFINITY)
                        {
                            if (std::isnan(lpt) || lpt != -INFINITY)
                            {
                                if (bad++ < 8)
                                    printf("FAIL %s nc=%d T=%g row=%d masked lpt=%g (want -inf)\n", mname, nc, T, m, lpt);
                                fail = true;
                            }
                        }
                        else
                        {
                            const double want_lpt = (double)ref_tlg - lse_ref[m];
                            if (!std::isfinite(lpt) || std::fabs((double)lpt - want_lpt) > 1e-4)
                            {
                                if (bad++ < 8)
                                    printf("FAIL %s nc=%d T=%g row=%d lpt=%g want=%g\n", mname, nc, T, m, lpt, want_lpt);
                                fail = true;
                            }
                        }
                    }
                }
                // Check TkRes (last seed) for NaN and exact top entries on row 0 (1 allowed) & row 4 (3 allowed across 2 ranks)
                for (int m = 0; m < M; ++m)
                    for (int k = 0; k < glue::LP_K; ++k)
                        if (std::isnan(tk0.lp[m][k]))
                        {
                            if (bad++ < 8) printf("FAIL %s nc=%d T=%g TkRes NaN row=%d k=%d\n", mname, nc, T, m, k);
                            fail = true;
                        }
                if (tk0.id[0][0] != u0 || std::fabs(tk0.lp[0][0]) > 1e-4f || tk0.rank[0] != 1)
                {
                    printf("FAIL %s nc=%d T=%g TkRes row 0 got id=%d lp=%g rank=%d\n",
                           mname, nc, T, tk0.id[0][0], tk0.lp[0][0], tk0.rank[0]);
                    fail = true;
                }
                if (tk0.id[4][0] != u4a || tk0.id[4][1] != u4b || tk0.id[4][2] != u4c ||
                    !std::isfinite(tk0.lp[4][2]) || tk0.lp[4][3] != -INFINITY)
                {
                    printf("FAIL %s nc=%d T=%g TkRes row 4 got ids=(%d,%d,%d) lps=(%g,%g,%g,%g)\n",
                           mname, nc, T, tk0.id[4][0], tk0.id[4][1], tk0.id[4][2],
                           tk0.lp[4][0], tk0.lp[4][1], tk0.lp[4][2], tk0.lp[4][3]);
                    fail = true;
                }
                printf("NR=%d mode=%-9s nc=%2d T=%3.1f: %s\n", NR, mname, nc, T, bad == 0 ? "PASS" : "FAIL");
            }

            for (int r = 0; r < NR; ++r)
            {
                cudaFree(d_lg[r]); cudaFree(d_send[r]); cudaFree(d_tkt[r]);
                cudaFree(out[r]); cudaFree(lpo[r]); cudaFree(lpto[r]); cudaFree(tko[r]);
            }
            cudaFree(recv); cudaFree(stepp); cudaFree(err); cudaFree(bell); cudaFree(miss); cudaFree(rows_d);
        }
    }

    cudaFree(tok_d); cudaFree(tgt_d);
    printf("\n=== head_mask_test (NR=%d): %s ===\n", NR, fail ? "FAIL" : "ALL PASS");
    return fail ? 1 : 0;
}
