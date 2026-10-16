// GPU test of the Flash DSA attention (flash_attn.cu + flash_bind.cu: NoPE MLA + kpool indexer) against the float64
// references of engine/ref/flash/dsa_inputs.py (real GLM 5.3 Flash layer-3 projections of real text).
// For each rank of G6_NR (head group (rank - layer) % NR) and each row count M: one binding (capacity = the run's
// tokens), rows fed in calls of M consecutive positions 0..N-1 (the last call may be shorter). At each checkpoint context
// C the call holding position C-1 is checked, row C-1:
//   selection  the row's indices (-1 dropped, sorted) == refa_C_sel. A differing pool passes only as a tie: its reference
//              score within TIE (relative) of the cut score (the reference's k-th best). Tail tokens must match exactly.
//   scores     the kernel's pool scores (g6_attn_index_scores) vs refa_C_score over the row's visible pools (reported)
//   output     FP16 [h][256] of the rank's heads vs refa_C_o: rel-L2 <= GATE (bf16-activation op gate);
//              vs refb (fp32-input reference) reported
//   replay     (--replay) the same call through g6_attn_bind_replay_init + g6_attn_replay_run: indices and output
//              bit-identical to the plain run
//   error word zero after every checked call
// Build (from engine/attn/tests), BF16 KV rows and FP8 KV rows:
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG6_NR=4 \
//        -o flash_dsa_test flash_dsa_test.cu ../flash_attn.cu ../flash_bind.cu
//   (same + -DG6_FLASH_KV_FP8=1 -o flash_dsa_test_fp8)
// Run: flash_dsa_test DIR [--m 1,8] [--ranks 0,1,2,3] [--ctx 1,5,2049,9000] [--gate 1e-2] [--replay]
//   DIR = dsa_inputs.py --out (meta.json, raw arrays). Prints one line per check, PASS/FAIL; exit 1 on FAIL.
#include "../attn.h"
#include "../bind.h"
#include "../replay.h"
#include "../prefill.h"
#include "../kvfmt.h"
#include "../../g6nr.h"
#include "../../model_geom.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <functional>
#include <iterator>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
extern "C" void* g6_attn_bind_prefill_output(const G6AttnPrefill*, const G6AttnBinding*, int rows);
extern "C" int* g6_attn_bind_prefill_indices(const G6AttnPrefill*, const G6AttnBinding*, int rows);
#define CK(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static std::string D;
template <class T> static std::vector<T> load(const std::string& name, size_t expect = 0)
{
    const std::string p = D + "/" + name;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END);
    const size_t n = ftell(f) / sizeof(T);
    fseek(f, 0, SEEK_SET);
    std::vector<T> v(n);
    if (fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "short read %s\n", p.c_str()); exit(2); }
    fclose(f);
    if (expect && n != expect) { fprintf(stderr, "%s: %zu elements, expected %zu\n", p.c_str(), n, expect); exit(2); }
    return v;
}
template <class T> static T* up(const std::vector<T>& v)
{
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static std::vector<int> ints(const char* s)
{
    std::vector<int> v;
    for (const char* p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return v;
}
static double rel_l2(const float* a, const float* b, size_t n)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { const double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return std::sqrt(num / (den > 0 ? den : 1e-300));
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s DIR [--m 1,8] [--ranks 0,1,2,3] [--ctx 1,5,2049,9000] [--gate 1e-2] [--replay] [--prefill]\n", argv[0]); return 2; }
    D = argv[1];
    std::vector<int> Ms = {1, 8}, ranks, ctxs = {1, 5, 2049, 9000};
    for (int r = 0; r < g6nr::NR; ++r) ranks.push_back(r);
    double GATE = 1e-2, TIE = 1e-5;
    bool replay = false, prefill = false;
    int L=3;
    for (int i = 2; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--layer") && i + 1 < argc) L=atoi(argv[++i]);
        else if (!strcmp(argv[i], "--m") && i + 1 < argc) Ms = ints(argv[++i]);
        else if (!strcmp(argv[i], "--ranks") && i + 1 < argc) ranks = ints(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctxs = ints(argv[++i]);
        else if (!strcmp(argv[i], "--gate") && i + 1 < argc) GATE = atof(argv[++i]);
        else if (!strcmp(argv[i], "--replay")) replay = true;
        else if (!strcmp(argv[i], "--prefill")) prefill = true;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if(L<0 || L>=geom::NLAYERS || geom::is_kda(L)) { fprintf(stderr,"not a DSA layer\n"); return 2; }
    const int H = geom::HEADS, NOPE = geom::QK_NOPE, LAT = geom::KV_LORA, VD = geom::V_HEAD;
    const int IH = g6kv::IDX_HEADS, ID = g6kv::IDX_DIM, SEL = g6kv::SEL, KP = g6kv::KPOOL;
    const auto kv_a = load<__half>("kv_a.f16");
    const int N = (int)(kv_a.size() / LAT);
    const auto q_b = load<__half>("q_b.f16", (size_t)N * H * NOPE);
    const auto idx_q = load<__half>("idx_q.f16", (size_t)N * IH * ID), idx_k = load<__half>("idx_k.f16", (size_t)N * ID);
    const auto idx_w = load<__half>("idx_w.f16", (size_t)N * IH), idx_g = load<__half>("idx_g.f16", (size_t)N * ID);
    const auto uk = load<uint16_t>("uk.bf16", (size_t)H * NOPE * LAT), uv = load<uint16_t>("uv.bf16", (size_t)H * VD * LAT);
    const auto kv_norm = load<uint16_t>("kv_norm.bf16", LAT), knw = load<uint16_t>("k_norm_w.bf16", ID);
    const auto knb = load<uint16_t>("k_norm_b.bf16", ID), ape = load<uint16_t>("ape.bf16", KP * ID);
    for (int C : ctxs) if (C < 1 || C > N) { fprintf(stderr, "context %d outside 1..%d\n", C, N); return 2; }
    printf("# %s: N=%d tokens, layer %d, NR=%d, KV row %s (%d B), M {", D.c_str(), N, L, g6nr::NR, g6kv::NAME, g6kv::ROW);
    for (int M : Ms) printf(" %d", M);
    printf(" }, gate %.1e, tie %.0e%s%s\n", GATE, TIE, replay ? ", replay" : "", prefill ? ", prefill" : "");

    struct Ref { std::vector<int> sel; std::vector<float> score, oa, ob; };
    std::vector<Ref> ref(ctxs.size());
    for (size_t c = 0; c < ctxs.size(); ++c)
    {
        const std::string t = std::to_string(ctxs[c]);
        ref[c].sel = load<int>("refa_" + t + "_sel.i32");
        ref[c].score = load<float>("refa_" + t + "_score.f32", (size_t)(ctxs[c] + KP - 1) / KP);
        ref[c].oa = load<float>("refa_" + t + "_o.f32", (size_t)H * VD);
        ref[c].ob = load<float>("refb_" + t + "_o.f32", (size_t)H * VD);
    }
    std::vector<int> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = i;
    int* dpos = up(pos);
    __half *dkv = up(kv_a), *diq = up(idx_q), *dik = up(idx_k), *diw = up(idx_w), *dig = up(idx_g);
    uint16_t *dnorm = up(kv_norm), *dknw = up(knw), *dknb = up(knb), *dape = up(ape);
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    G6AttnStepCell* dcell;
    CK(cudaMalloc(&dcell, sizeof(G6AttnStepCell)));
    int fails = 0, checks = 0;

    for (int rank : ranks)
    {
        const int group = (rank - L % g6nr::NR + g6nr::NR) % g6nr::NR;
        const int h = g6nr::width(H, group), h0 = g6nr::start(H, group);
        // the rank's heads: q_b rows [N][h][256], uk / uv [h][256][512] (head-major, contiguous)
        std::vector<__half> qr((size_t)N * h * NOPE);
        for (int t = 0; t < N; ++t)
            memcpy(&qr[(size_t)t * h * NOPE], &q_b[((size_t)t * H + h0) * NOPE], (size_t)h * NOPE * 2);
        std::vector<uint16_t> ukr(uk.begin() + (size_t)h0 * NOPE * LAT, uk.begin() + (size_t)(h0 + h) * NOPE * LAT);
        std::vector<uint16_t> uvr(uv.begin() + (size_t)h0 * VD * LAT, uv.begin() + (size_t)(h0 + h) * VD * LAT);
        __half* dq = up(qr);
        uint16_t *duk = up(ukr), *duv = up(uvr);
        auto T = [](const void* p, std::initializer_list<int64_t> s) {
            G6AttnTensor t{}; t.data = p; t.ndim = (int)s.size();
            int j = 0; for (auto d : s) t.shape[j++] = d;
            return t;
        };
        G6AttnPack pack{T(duk, {h, NOPE, LAT}), T(duv, {h, VD, LAT}), T(dnorm, {LAT}), T(dknw, {ID}), T(dknb, {ID})};
        pack.index_ape = T(dape, {KP, ID});
        for (int M : Ms)
        {
            G6AttnBinding b{};
            const int bind_maxr = prefill ? 8 : std::max(M, 8);
            const int maxr = std::max(M, 8);
            if (int e = g6_attn_bind_create_ext(&b, &pack, rank, L, N, bind_maxr, prefill ? 8 : 16, st, nullptr, 0))
            { fprintf(stderr, "rank %d: bind_create failed %d\n", rank, e); return 2; }
            G6AttnPrefill* pf = nullptr;
            if (prefill)
            {
                if (int e = g6_attn_bind_prefill_create(&pf, &b, st))
                { fprintf(stderr, "rank %d: prefill_create failed %d\n", rank, e); return 2; }
            }
            if (b.args.heads != h || b.head_begin != h0) { fprintf(stderr, "rank %d: head group %d+%d, binder %d+%d\n", rank, h0, h, b.head_begin, b.args.heads); return 2; }
            float* dscores;
            const int pst = ((N + KP - 1) / KP + 31) / 32 * 32;
            CK(cudaMalloc(&dscores, (size_t)maxr * pst * 4));
            std::vector<int> ind((size_t)maxr * SEL), ind2((size_t)maxr * SEL);
            std::vector<__half> out((size_t)maxr * h * VD), out2((size_t)maxr * h * VD);
            std::vector<float> sc((size_t)maxr * pst);
            size_t next = 0;
            std::vector<int> order(ctxs.size());
            for (size_t c = 0; c < ctxs.size(); ++c) order[c] = (int)c;
            std::sort(order.begin(), order.end(), [&](int a, int b2) { return ctxs[a] < ctxs[b2]; });
            for (int r0 = 0; r0 < N && next < order.size(); r0 += M)
            {
                const int n = std::min(M, N - r0), context = r0 + n;
                G6AttnStep s{};
                s.rows = n; s.context = context; s.positions = dpos + r0;
                s.q_b = dq + (size_t)r0 * h * NOPE; s.kv_a = dkv + (size_t)r0 * LAT;
                s.index_q = diq + (size_t)r0 * IH * ID; s.index_k = dik + (size_t)r0 * ID;
                s.index_weight = diw + (size_t)r0 * IH; s.index_gate = dig + (size_t)r0 * ID;
                const int rc = prefill ? g6_attn_bind_prefill_run(pf, &b, &s, st) : g6_attn_bind_run(&b, &s, st);
                if (rc) { fprintf(stderr, "rank %d M %d row %d: run failed %d\n", rank, M, r0, rc); return 2; }
                while (next < order.size() && ctxs[order[next]] <= context)
                {
                    const int ci = order[next++], C = ctxs[ci], row = C - 1 - r0;
                    const Ref& R = ref[ci];
                    unsigned err = 0;
                    const int* dind = prefill ? g6_attn_bind_prefill_indices(pf, &b, n) : b.args.indices;
                    const void* dout = prefill ? g6_attn_bind_prefill_output(pf, &b, n) : b.args.output;
                    CK(cudaMemcpyAsync(&err, b.args.error, 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(ind.data(), dind, (size_t)n * SEL * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(out.data(), dout, (size_t)n * h * VD * 2, cudaMemcpyDeviceToHost, st));
                    double sdiff = 0;
                    if (!prefill)
                    {
                        if (int e = g6_attn_index_scores(&b.args, b.scratch, b.scratch_bytes, dscores, st)) { fprintf(stderr, "index_scores failed %d\n", e); return 2; }
                        const int ps = ((context + KP - 1) / KP + 31) / 32 * 32;
                        CK(cudaMemcpyAsync(sc.data(), dscores, (size_t)n * ps * 4, cudaMemcpyDeviceToHost, st));
                    }
                    CK(cudaStreamSynchronize(st));
                    // selection
                    std::vector<int> got;
                    for (int j = 0; j < SEL; ++j) if (ind[(size_t)row * SEL + j] >= 0) got.push_back(ind[(size_t)row * SEL + j]);
                    std::sort(got.begin(), got.end());
                    const bool dup = std::adjacent_find(got.begin(), got.end()) != got.end();
                    const int full = (C / KP) * KP;               // tokens in complete pools; the tail C % 4 always attended
                    std::vector<int> gp, rp;
                    bool tail_ok = true;
                    auto split = [&](const std::vector<int>& v, std::vector<int>& pools, int& tail) {
                        tail = 0;
                        for (size_t j = 0; j < v.size(); ++j)
                        {
                            if (v[j] >= full) { ++tail; continue; }
                            if (v[j] % KP == 0) pools.push_back(v[j] / KP);
                        }
                    };
                    int gt, rt;
                    split(got, gp, gt); split(R.sel, rp, rt);
                    for (int t = full; t < C; ++t)
                        if (!std::binary_search(got.begin(), got.end(), t)) tail_ok = false;
                    for (int t : got) if (t >= C) tail_ok = false;
                    // every selected pool must be whole: 4 tokens of it in got
                    bool whole = true;
                    for (int p : gp) for (int j = 0; j < KP; ++j) if (!std::binary_search(got.begin(), got.end(), p * KP + j)) whole = false;
                    if ((int)got.size() != (int)gp.size() * KP + gt) whole = false;
                    std::vector<int> diff;
                    std::set_symmetric_difference(gp.begin(), gp.end(), rp.begin(), rp.end(), std::back_inserter(diff));
                    // the cut: the k-th best visible reference score
                    std::vector<float> vis;
                    float amax = 0;
                    for (size_t p = 0; p < R.score.size(); ++p) if (std::isfinite(R.score[p])) { vis.push_back(R.score[p]); amax = std::max(amax, std::fabs(R.score[p])); }
                    std::sort(vis.begin(), vis.end(), std::greater<float>());
                    const int k = std::min((int)vis.size(), g6kv::TOP_POOLS);
                    const float cut = k ? vis[k - 1] : 0.f;
                    double worst_tie = 0;
                    bool ties_ok = true;
                    for (int p : diff)
                    {
                        const double g = std::fabs((double)R.score[p] - cut) / std::max(1e-30, (double)amax);
                        worst_tie = std::max(worst_tie, g);
                        if (!std::isfinite(R.score[p]) || g > TIE) ties_ok = false;
                    }
                    // pool scores over the visible pools
                    if (!prefill)
                    {
                        const int ps = ((context + KP - 1) / KP + 31) / 32 * 32;
                        for (size_t p = 0; p < R.score.size(); ++p)
                            if (std::isfinite(R.score[p])) sdiff = std::max(sdiff, std::fabs((double)sc[(size_t)row * ps + p] - R.score[p]));
                        sdiff /= std::max(1e-30, (double)amax);
                    }
                    // output
                    std::vector<float> o((size_t)h * VD);
                    for (size_t j = 0; j < o.size(); ++j) o[j] = __half2float(out[(size_t)row * h * VD + j]);
                    const double ea = rel_l2(o.data(), R.oa.data() + (size_t)h0 * VD, o.size());
                    const double eb = rel_l2(o.data(), R.ob.data() + (size_t)h0 * VD, o.size());
                    bool rp_ok = true;
                    if (replay && !prefill)
                    {
                        G6AttnReplay rpl{};
                        if (int e = g6_attn_bind_replay_init(&b, &s, dcell, N, &rpl, st)) { fprintf(stderr, "replay_init failed %d\n", e); return 2; }
                        G6AttnStepCell cell{};
                        cell.context = context; cell.page_count = cell.table_size = (context + g6kv::PAGE - 1) / g6kv::PAGE;
                        CK(cudaMemcpyAsync(dcell, &cell, sizeof cell, cudaMemcpyHostToDevice, st));
                        if (int e = g6_attn_replay_run(&rpl, st)) { fprintf(stderr, "replay_run failed %d\n", e); return 2; }
                        unsigned err2 = 0;
                        CK(cudaMemcpyAsync(&err2, b.args.error, 4, cudaMemcpyDeviceToHost, st));
                        CK(cudaMemcpyAsync(ind2.data(), b.args.indices, (size_t)n * SEL * 4, cudaMemcpyDeviceToHost, st));
                        CK(cudaMemcpyAsync(out2.data(), b.args.output, (size_t)n * h * VD * 2, cudaMemcpyDeviceToHost, st));
                        CK(cudaStreamSynchronize(st));
                        rp_ok = !err2 && !memcmp(ind.data(), ind2.data(), (size_t)n * SEL * 4) && !memcmp(out.data(), out2.data(), (size_t)n * h * VD * 2);
                    }
                    const bool sel_ok = !dup && whole && tail_ok && ties_ok && rt == gt;
                    const bool ok = !err && sel_ok && ea <= GATE && std::isfinite(ea) && rp_ok;
                    ++checks; fails += !ok;
                    printf("%s rank %d heads %2d+%2d M %d C %5d: err %u, selected %4zu (ref %4zu, %zu pool(s) differ, worst tie gap %.1e), "
                           "pool scores max|d| %.1e rel, o rel-L2 %.2e (fp32-input ref %.2e)%s\n",
                           ok ? "PASS" : "FAIL", rank, h0, h, M, C, err, got.size(), R.sel.size(), diff.size(), worst_tie, sdiff, ea, eb,
                           replay ? (rp_ok ? ", replay bit-identical" : ", REPLAY DIFFERS") : "");
                    if (!sel_ok) printf("     selection: dup %d whole %d tail %d (tail tokens got %d ref %d) ties %d\n", dup, whole, tail_ok, gt, rt, ties_ok);
                }
            }
            CK(cudaFree(dscores));
            if (pf) g6_attn_bind_prefill_release(pf);
            g6_attn_bind_release(&b);
        }
        CK(cudaFree(dq)); CK(cudaFree(duk)); CK(cudaFree(duv));
    }
    printf("%s: %d checks, %d failed\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
