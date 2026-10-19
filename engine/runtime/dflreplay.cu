// dflreplay: the G6_DFL_BODY_I4 judge. A G6_DFL_DUMP (rank 0 of a --dflash run, dflash.cuh dfl_dump_step) replayed step
// by step through two DFlash2 drafters on ONE GPU, all NR ranks of each with a loopback fabric (as dfltest): A on the
// MXFP8 body records, B on the INT4 g32 copies G6_DFL_BODY_I4 makes (dflash.cuh dfl_body_i4, the same function). Both get
// the dump's context features, positions and bonus tokens, so every draft step sees the same context in A and B (the
// target's trajectory is the dump's; matched, unlike two live runs whose acceptance changes what follows).
//
// dflreplay DIR DUMP [--sel 0|1] [--head-i4 0|1] [--temp T] [--seed S] [--top-p P] [--rounds N] [--bench N] [--out FILE]
//   DIR/dflash.r<r>.manifest.json + .pack   drafter packs (tools/convert/dflash2.py), as dfltest
//   DIR/head.r<r>.manifest.json + .pack     the target lm_head slice of rank r (dflash2.py --head), as dfltest
//   DIR/emb.bin   int32 n, int32 ids[n], float32 rows[n][6144]: the target embedding rows of the dump's bonus tokens and
//                 the mask token (engine/ref/dflreplay_emb.py writes it)
//   --sel        default: the pack's (DFlash2 selector walk when it has one)
//   --head-i4    1 (default, as serving with G6_DFL_HEAD_I4=1): both drafters' heads read INT4 records; 0: MXFP8
//   --temp / --seed / --top-p   the drafts' sampling keys (the dump does not record them; 0 = greedy, the drafter's own
//                default under rejection sampling). Use the dumping run's values for the dump column to match A.
//   --bench N    then time one rank's body GEMVs of a draft step (context fc + 6 kv, block 6 x aconv q kv o mconv gu
//                down) for A and B as a captured graph, N replays, cold-ish (the 700 MB set does not fit the 25 MB L2)
//   --out FILE   one TSV line per draft step: q0, accepted (dump, A, B), committed tokens, drafts and lp of A and B, then the
//                lp bits of A and B (hex)
// Printed: draft steps scored against the committed tokens; per position r = 1..7 the share of steps where A and B
// draft the same token; accepted-length histograms; tokens per round 1 + mean(min(accepted, k)) for k = 1..7, which
// with the verify table (ms at M = k + 1) and the drafter's step time gives committed tokens per ms; A == dump (a check of
// the replay: equal when the dump's run had the same head and sampling).
#include <cstdio>
#include <map>
#include <vector>
#include "packfind.h"
#include "dflash.cuh"

using namespace g6rt;
constexpr int NR = g6nr::NR;

struct DLoop { const char* send[NR]; char* recv; const u64* stepp; int c, ncoll, n16; size_t stride; };
// the NIC: every rank's send slot of collective c -> the shared recv slot [c][rank], then the tail. grid NR, block 128.
__global__ void __launch_bounds__(128) k_dloop(DLoop a)
{
    const u64 step = *a.stepp;
    const size_t slot = (step & 1) * a.ncoll + a.c;
    const uint4* s = reinterpret_cast<const uint4*>(a.send[blockIdx.x] + slot * a.stride);
    uint4* d = reinterpret_cast<uint4*>(a.recv + (slot * NR + blockIdx.x) * a.stride);
    for (int i = threadIdx.x; i < a.n16; i += 128) d[i] = s[i];
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); *reinterpret_cast<u64*>(d + a.n16) = g6::coll_seq(step, a.c, a.ncoll); }
}

// one drafter (all ranks) with its own loopback fabric
struct Var
{
    std::vector<Dfl> D;
    std::vector<DflOut*> out;
    DLoop loop{};
    unsigned* err = nullptr;
};
static void build_var(Var& v, std::vector<Model>& M, const std::vector<Find>& fd, const DflMeta& meta, bool sel, const DflIn* in,
                      const int* rows_d, const std::vector<const uint8_t*>& hi4, bool body_i4, cudaStream_t st)
{
    const size_t stride = ((size_t)dfl::B * HID * 2 + 8 + 255) / 256 * 256;
    char* recv = dmalloc<char>(2 * (size_t)dfl::NC * NR * stride);
    u64* stepp = dmalloc<u64>(1);
    v.err = dmalloc<unsigned>(1);
    unsigned long long* bell = dmalloc<unsigned long long>(dfl::NC);
    v.D.resize(NR);
    v.out.resize(NR);
    for (int r = 0; r < NR; ++r)
    {
        g6::FabricDev f{};
        f.send = dmalloc<char>(2 * (size_t)dfl::NC * stride);
        f.recv = recv; f.doorbell = bell; f.error = v.err;
        f.rank = r; f.nr = NR; f.ncoll = dfl::NC; f.stride = stride;
        v.loop.send[r] = f.send;
        v.out[r] = dmalloc<DflOut>(1);
        Dfl& D = v.D[r];
        D.in = in; D.out = v.out[r]; D.nc = 0;
        dfl_build(D, M[r], fd[r], meta, f, stepp, sel, st);
        D.emb_rows = rows_d;          // block row r reads row rows_d[r] of the replay's embedding table
        if (hi4[r]) dfl_head_i4(D, hi4[r], 1, 32);
        if (body_i4) dfl_body_i4(D, st);
    }
    v.loop.recv = recv; v.loop.stepp = stepp; v.loop.ncoll = dfl::NC; v.loop.stride = stride;
}
// one step of every rank (DflIn already in device memory; feat already set)
static void run_var(Var& v, bool query, cudaStream_t st)
{
    for (int p = 0; p < (query ? dfl::NPH : 2); ++p)
    {
        for (Dfl& x : v.D) dfl_phase(x, p, st, query);
        if (dfl_delivered(p, query))
        {
            DLoop a = v.loop;
            a.c = p; a.n16 = (int)(dfl_payload(v.D[0], p) / 16);
            k_dloop<<<NR, 128, 0, st>>>(a);
        }
    }
}
static DflOut out_of(Var& v, const char* name)
{
    unsigned e = 0;
    G6RT_CK(cudaMemcpy(&e, v.err, 4, cudaMemcpyDeviceToHost));
    if (e) die(std::string(name) + ": fabric error word " + std::to_string(e));
    std::vector<DflOut> o(NR);
    for (int r = 0; r < NR; ++r) G6RT_CK(cudaMemcpy(&o[r], v.out[r], sizeof(DflOut), cudaMemcpyDeviceToHost));
    for (int r = 1; r < NR; ++r)
        if (memcmp(o[r].tok + 1, o[0].tok + 1, sizeof(int) * dfl::K) || memcmp(o[r].lp + 1, o[0].lp + 1, sizeof(float) * dfl::K))
            die(std::string(name) + ": ranks disagree on the drafts or their log-probabilities");
    return o[0];
}
// rank 0's body GEMVs of one draft step, captured, timed over n replays -> ms per step
static double bench_var(Var& v, int n, cudaStream_t st)
{
    Dfl& D = v.D[0];
    cudaGraph_t g;
    cudaGraphExec_t ge;
    G6RT_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    bind::launch(D.fc, st);
    for (const DLayer& y : D.ly) bind::launch(y.kvc, st);
    for (const DLayer& y : D.ly)
    {
        bind::launch(y.ac, st); bind::launch(y.q, st); bind::launch(y.kv, st);
        if constexpr (dfl::REP) for (int k = 0; k < dfl::KS; ++k) bind::launch(y.ok[k], st);
        else bind::launch(y.o, st);
        bind::launch(y.mc, st); bind::launch(y.gu, st);
        if constexpr (dfl::REP) for (int k = 0; k < dfl::KS; ++k) bind::launch(y.dk[k], st);
        else bind::launch(y.down, st);
    }
    G6RT_CK(cudaStreamEndCapture(st, &g));
    G6RT_CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 3; ++i) G6RT_CK(cudaGraphLaunch(ge, st));
    cudaEvent_t a, b;
    G6RT_CK(cudaEventCreate(&a));
    G6RT_CK(cudaEventCreate(&b));
    G6RT_CK(cudaEventRecord(a, st));
    for (int i = 0; i < n; ++i) G6RT_CK(cudaGraphLaunch(ge, st));
    G6RT_CK(cudaEventRecord(b, st));
    G6RT_CK(cudaEventSynchronize(b));
    float ms = 0;
    G6RT_CK(cudaEventElapsedTime(&ms, a, b));
    G6RT_CK(cudaGraphExecDestroy(ge));
    G6RT_CK(cudaGraphDestroy(g));
    return ms / n;
}

struct Round { int q0, tok0, dump[dfl::B]; DflOut a, b; };
static int accept(const int* d, const std::vector<int>& toks, int q0)
{
    int n = 0;
    while (n < dfl::K && q0 + 1 + n < (int)toks.size() && d[1 + n] == toks[q0 + 1 + n]) ++n;
    return n;
}

int main(int argc, char** argv)
{
    if (argc < 3) die("usage: dflreplay DIR DUMP [--sel 0|1] [--head-i4 0|1] [--temp T] [--seed S] [--top-p P] [--rounds N] [--bench N] [--out FILE]");
    const std::string dir = argv[1], dump = argv[2];
    int sel = -1, hi4 = 1, max_rounds = 0, bench = 0;
    float temp = 0.f, top_p = 0.f;
    uint32_t seed = 1234;
    std::string outp;
    for (int i = 3; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--sel") sel = atoi(argv[i + 1]);
        else if (a == "--head-i4") hi4 = atoi(argv[i + 1]);
        else if (a == "--temp") temp = (float)atof(argv[i + 1]);
        else if (a == "--seed") seed = (uint32_t)strtoul(argv[i + 1], nullptr, 10);
        else if (a == "--top-p") top_p = (float)atof(argv[i + 1]);
        else if (a == "--rounds") max_rounds = atoi(argv[i + 1]);
        else if (a == "--bench") bench = atoi(argv[i + 1]);
        else if (a == "--out") outp = argv[i + 1];
        else die("unknown argument " + a);
    }
    // the embedding rows
    FILE* ef = fopen((dir + "/emb.bin").c_str(), "rb");
    if (!ef) die("no " + dir + "/emb.bin (engine/ref/dflreplay_emb.py)");
    int ne = 0;
    if (fread(&ne, 4, 1, ef) != 1 || ne < 1) die("emb.bin: bad header");
    std::vector<int> eid(ne);
    std::vector<float> erow((size_t)ne * HID);
    if (fread(eid.data(), 4, ne, ef) != (size_t)ne || fread(erow.data(), 4, erow.size(), ef) != erow.size()) die("emb.bin: short");
    fclose(ef);
    std::map<int, int> erix;
    for (int i = 0; i < ne; ++i) erix[eid[i]] = i;
    std::vector<__nv_bfloat16> eb(erow.size());
    for (size_t i = 0; i < erow.size(); ++i) eb[i] = __float2bfloat16(erow[i]);
    __nv_bfloat16* emb_d = dmalloc<__nv_bfloat16>(eb.size());
    G6RT_CK(cudaMemcpy(emb_d, eb.data(), eb.size() * 2, cudaMemcpyHostToDevice));
    int* rows_d = dmalloc<int>(dfl::B);

    G6RT_CK(bind::init());
    cudaStream_t st;
    G6RT_CK(cudaStreamCreate(&st));
    std::vector<g6::Pack> dp(NR), hp(NR);
    DevStore store;
    std::vector<Model> M(NR);
    std::vector<Find> fd;
    std::vector<const uint8_t*> h4(NR, nullptr);
    DflMeta meta;
    for (int r = 0; r < NR; ++r)
    {
        const std::string b = dir + "/dflash.r" + std::to_string(r), h = dir + "/head.r" + std::to_string(r);
        load_pack(dp[r], b + ".manifest.json", b + ".pack", r);
        load_pack(hp[r], h + ".manifest.json", h + ".pack", r);
        fd.push_back(pack_find(dp[r], store));
        const Find fh = pack_find(hp[r], store);
        Model& m = M[r];
        m.rank = r;
        m.emb = emb_d;
        m.head_t = bind::head_tensors([&](const char* nm) { return as_bind(fh(nm)); });
        if (!bind::vocab_slice(r, m.vlo, m.vn)) die("vocab slice");
        if (hi4) h4[r] = head_i4_records(m.head_t.lm_head, m.vn, 32, r, "dflreplay head", st);
        if (r == 0) meta = dfl_read_meta(fd[0]);
    }
    if (erix.find(meta.mask_id) == erix.end()) die("emb.bin has no row for the mask token " + std::to_string(meta.mask_id));
    const bool use_sel = sel < 0 ? meta.sel : sel == 1;
    DflIn* in = dmalloc<DflIn>(1);
    Var A, B;
    build_var(A, M, fd, meta, use_sel, in, rows_d, h4, false, st);
    build_var(B, M, fd, meta, use_sel, in, rows_d, h4, true, st);
    fprintf(stderr, "# device tensors: %.2f GB held, %.2f GB shared between ranks; %s, head %s\n", store.bytes / 1e9, store.shared_bytes / 1e9,
            use_sel ? "selector walk" : "argmax", hi4 ? "INT4" : "MXFP8");

    FILE* df = fopen(dump.c_str(), "rb");
    if (!df) die("cannot open " + dump);
    FILE* of = outp.empty() ? nullptr : fopen(outp.c_str(), "w");
    if (of) fprintf(of, "q0\tacc_dump\tacc_a\tacc_b\tcommitted\tdrafts_a\tdrafts_b\tlp_a\tlp_b\tlpbits_a\tlpbits_b\n");
    std::vector<Round> req;               // the request's draft steps, scored at its token record
    long n_steps = 0, n_rounds = 0, same_pos[dfl::B] = {}, same_all = 0, a_eq_dump = 0;
    long hist_d[dfl::B] = {}, hist_a[dfl::B] = {}, hist_b[dfl::B] = {};
    int lo = -1;
    u64 step = 0;
    std::vector<__half> fh((size_t)dfl::B * dfl::NT * HID);
    bool stop = false;
    while (!stop)
    {
        int h[12];
        if (fread(h, 4, 12, df) != 12) break;
        if (h[0] == 2)                    // the request's committed tokens: score its draft steps
        {
            std::vector<int> toks(h[1]);
            if (fread(toks.data(), 4, toks.size(), df) != toks.size()) die("dump: short token record");
            for (const Round& x : req)
            {
                const int ad = accept(x.dump, toks, x.q0), aa = accept(x.a.tok, toks, x.q0), ab = accept(x.b.tok, toks, x.q0);
                ++hist_d[ad]; ++hist_a[aa]; ++hist_b[ab]; ++n_rounds;
                bool all = true;
                for (int r = 1; r < dfl::B; ++r)
                {
                    const bool s = x.a.tok[r] == x.b.tok[r];
                    same_pos[r] += s;
                    all &= s;
                }
                same_all += all;
                a_eq_dump += !memcmp(x.a.tok + 1, x.dump + 1, sizeof(int) * dfl::K);
                if (of)
                {
                    fprintf(of, "%d\t%d\t%d\t%d\t", x.q0, ad, aa, ab);
                    for (int r = 1; r < dfl::B; ++r) fprintf(of, "%s%d", r > 1 ? "," : "", x.q0 + r < (int)toks.size() ? toks[x.q0 + r] : -1);
                    for (const DflOut* o : {&x.a, &x.b})
                    {
                        fputc('\t', of);
                        for (int r = 1; r < dfl::B; ++r) fprintf(of, "%s%d", r > 1 ? "," : "", o->tok[r]);
                    }
                    for (const DflOut* o : {&x.a, &x.b})
                    {
                        fputc('\t', of);
                        for (int r = 1; r < dfl::B; ++r) fprintf(of, "%s%.4f", r > 1 ? "," : "", o->lp[r]);
                    }
                    for (const DflOut* o : {&x.a, &x.b})   // the lp bits, for a byte comparison between builds (TP4 vs TP6)
                    {
                        fputc('\t', of);
                        for (int r = 1; r < dfl::B; ++r) { uint32_t u; memcpy(&u, &o->lp[r], 4); fprintf(of, "%s%08x", r > 1 ? "," : "", u); }
                    }
                    fputc('\n', of);
                }
            }
            req.clear();
            lo = -1;
            continue;
        }
        const int kind = h[0], pos0 = h[1], rows = h[2], q0 = h[3], tok0 = h[4];
        if (rows < 1 || rows > dfl::B) die("dump: bad context row count " + std::to_string(rows));
        std::fill(fh.begin(), fh.end(), __float2half(0.f));
        if (fread(fh.data(), 2, (size_t)rows * dfl::NT * HID, df) != (size_t)rows * dfl::NT * HID) die("dump: short features");
        if (lo < 0) lo = pos0;            // the first context row of a request is its lowest position
        const bool query = kind == 1;
        for (Var* v : {&A, &B})
            for (Dfl& x : v->D) G6RT_CK(cudaMemcpyAsync(x.feat, fh.data(), fh.size() * 2, cudaMemcpyHostToDevice, st));
        if (query)
        {
            if (erix.find(tok0) == erix.end()) die("emb.bin has no row for bonus token " + std::to_string(tok0));
            int er[dfl::B];
            er[0] = erix[tok0];
            for (int r = 1; r < dfl::B; ++r) er[r] = erix[meta.mask_id];
            G6RT_CK(cudaMemcpyAsync(rows_d, er, sizeof er, cudaMemcpyHostToDevice, st));
        }
        DflIn di;
        memset(&di, 0, sizeof di);
        di.step = ++step;
        di.ctx_rows = rows; di.ctx_pos0 = pos0; di.q0 = q0; di.lo = lo; di.tok0 = tok0; di.query = query;
        for (int r = 0; r < dfl::B; ++r) di.srow[r] = glue::SampleRow{temp > 0 ? 1.f / temp : 0.f, seed, 0u, q0 + r, 0, top_p};
        G6RT_CK(cudaMemcpyAsync(in, &di, sizeof di, cudaMemcpyHostToDevice, st));
        run_var(A, query, st);
        run_var(B, query, st);
        G6RT_CK(cudaStreamSynchronize(st));
        G6RT_CK(cudaGetLastError());
        ++n_steps;
        if (query)
        {
            Round x;
            x.q0 = q0; x.tok0 = tok0;
            for (int r = 0; r < dfl::B; ++r) x.dump[r] = r ? h[4 + r] : tok0;
            x.a = out_of(A, "A (MXFP8)");
            x.b = out_of(B, "B (INT4)");
            req.push_back(x);
            if (max_rounds && n_rounds + (long)req.size() >= max_rounds) stop = true;
        }
    }
    fclose(df);
    if (of) fclose(of);
    if (!req.empty()) fprintf(stderr, "# %zu draft steps after the last token record: not scored\n", req.size());
    printf("replayed %ld drafter steps; scored %ld draft steps\n", n_steps, n_rounds);
    if (n_rounds)
    {
        const double n = (double)n_rounds;
        printf("A == dump drafts: %.4f\nA == B, all 7 drafts: %.4f\nA == B by position:", a_eq_dump / n, same_all / n);
        for (int r = 1; r < dfl::B; ++r) printf(" %.4f", same_pos[r] / n);
        printf("\naccepted histogram 0..7\n");
        for (auto [name, hh] : {std::pair<const char*, long*>{"dump", hist_d}, {"A", hist_a}, {"B", hist_b}})
        {
            printf("  %-4s", name);
            for (int a = 0; a < dfl::B; ++a) printf(" %ld", hh[a]);
            printf("\n");
        }
        printf("tokens per round at k drafts verified (1 + mean min(accepted, k)), k = 1..7\n");
        for (auto [name, hh] : {std::pair<const char*, long*>{"dump", hist_d}, {"A", hist_a}, {"B", hist_b}})
        {
            printf("  %-4s", name);
            for (int k = 1; k <= dfl::K; ++k)
            {
                double s = 0;
                for (int a = 0; a < dfl::B; ++a) s += (double)hh[a] * std::min(a, k);
                printf(" %.4f", 1 + s / n);
            }
            printf("\n");
        }
    }
    if (bench > 0)
    {
        const double ta = bench_var(A, bench, st), tb = bench_var(B, bench, st);
        size_t ba = 0, bb = 0;
        auto add = [](size_t& s, const Dfl& D) {
            s += bind::gv_bytes(D.fc);
            for (const DLayer& y : D.ly) s += bind::gv_bytes(y.kvc) + bind::gv_bytes(y.ac) + bind::gv_bytes(y.q) + bind::gv_bytes(y.kv) +
                                              bind::gv_bytes(y.o) + bind::gv_bytes(y.mc) + bind::gv_bytes(y.gu) + bind::gv_bytes(y.down);
            if constexpr (dfl::REP)
                for (const DLayer& y : D.ly)
                    for (int k = 0; k < dfl::KS; ++k) s += bind::gv_bytes(y.ok[k]) + bind::gv_bytes(y.dk[k]);
        };
        add(ba, A.D[0]);
        add(bb, B.D[0]);
        printf("bench rank 0 body GEMVs of a draft step (%d replays): A MXFP8 %.3f ms (%.1f MB, %.0f GB/s), B INT4 %.3f ms (%.1f MB, %.0f GB/s), B - A %.3f ms\n",
               bench, ta, ba / 1e6, ba / ta / 1e6, tb, bb / 1e6, bb / tb / 1e6, tb - ta);
    }
    return 0;
}
