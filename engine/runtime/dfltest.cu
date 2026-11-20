// dfltest: the DFlash2 drafter step (dflash.cuh) of all NR ranks on ONE GPU with a loopback fabric, for the reference
// check against engine/ref/dflash2_ref.py (engine/ref/dflash2_check.py writes the case and compares).
//
// dfltest DIR [--sel 0|1] [--temp T --seed S] [--fill 1]
//   DIR/dflash.r<r>.manifest.json + .pack   drafter packs (engine/loader-rs/src/dflash2.rs)
//   DIR/head.r<r>.manifest.json + .pack     the target lm_head slice of rank r, G.lm_head.mx.gv (dflash2.py --head)
//   DIR/case.bin   int32 n (context rows), p0 (first position), bonus (token), lo (first position the sequence wrote);
//                  float32 feat [n][NT x HID] (the target residual rows), float32 emb [8][HID] (embedding rows of the
//                  bonus and 7 mask tokens, which stand in for the 1.9 GB table)
//   DIR/out.<sel>.bin  int32 drafts [8] (rank 0; [0] unused), float32 x [8][HID] (rank 0's final-norm rows),
//                      float32 kv [L][2 (k, v)][NKV][n_last][128] (the ring at the last chunk's positions, every rank)
// The context rows go through context-only steps of 8 rows, the last chunk through the whole step. --fill 1: the context
// rows but the last chunk's go through fill steps instead (dfl_fill_phase, the served prompt-context path): each row
// in Dfl::pf_feat of one rank (row i on rank i % NR, zeros elsewhere, as the prefill's owner rows), gathered over the
// loopback, with collective NT (the fc slices; under REP the 0-byte barrier). Every output must equal the run without.
// Flash: pf_feat is written by the prefill's hook, dfl_pf_tap_flash, from streams with their mean on the rows a rank
// keeps and NaN on the others. With GR8_GLM53F_PF_SP=1 those are its owner rows, else every rank holds every row.
#include <cstdio>
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
__global__ void k_h2f(const __half* a, float* b, int n) { for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) b[i] = __half2float(a[i]); }

int main(int argc, char** argv)
{
    if (argc < 2) die("usage: dfltest DIR [--sel 0|1] [--temp T] [--seed S] [--fill 1]");
    const std::string dir = argv[1];
    int sel = 1;
    float temp = 0.f;
    uint32_t seed = 1234;
    bool fill = false;
    for (int i = 2; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--sel") sel = atoi(argv[i + 1]);
        else if (a == "--temp") temp = (float)atof(argv[i + 1]);
        else if (a == "--seed") seed = (uint32_t)strtoul(argv[i + 1], nullptr, 10);
        else if (a == "--fill") fill = atoi(argv[i + 1]) != 0;
        else die("unknown argument " + a);
    }
    FILE* cf = fopen((dir + "/case.bin").c_str(), "rb");
    if (!cf) die("no " + dir + "/case.bin");
    int hdr[4];
    if (fread(hdr, 4, 4, cf) != 4) die("case.bin: short header");
    const int n = hdr[0], p0 = hdr[1], bonus = hdr[2], lo = hdr[3];
    std::vector<float> feat((size_t)n * dfl::NT * HID), emb((size_t)dfl::B * HID);
    if (fread(feat.data(), 4, feat.size(), cf) != feat.size() || fread(emb.data(), 4, emb.size(), cf) != emb.size()) die("case.bin: short");
    fclose(cf);
    G6RT_CK(bind::init());
    cudaStream_t st;
    G6RT_CK(cudaStreamCreate(&st));
    const size_t stride = ((size_t)dfl::B * HID * 2 + 8 + 255) / 256 * 256;
    char* recv = dmalloc<char>(2 * (size_t)dfl::NC * NR * stride);
    u64* stepp = dmalloc<u64>(1);
    unsigned* err = dmalloc<unsigned>(1);
    unsigned long long* bell = dmalloc<unsigned long long>(dfl::NC);
    DflIn* in = dmalloc<DflIn>(1);
    std::vector<DflOut*> out(NR);
    std::vector<g6::Pack> dp(NR), hp(NR);
    DevStore store;
    std::vector<Model> M(NR);
    std::vector<Dfl> D(NR);
    DLoop loop{};
    // the embedding rows of the case, addressed by row: tokens 0..7 -> rows 0..7
    std::vector<__nv_bfloat16> eb(emb.size());
    for (size_t i = 0; i < emb.size(); ++i) eb[i] = __float2bfloat16(emb[i]);
    __nv_bfloat16* emb_d = dmalloc<__nv_bfloat16>(eb.size());
    G6RT_CK(cudaMemcpy(emb_d, eb.data(), eb.size() * 2, cudaMemcpyHostToDevice));
    int rows[dfl::B];
    for (int r = 0; r < dfl::B; ++r) rows[r] = r;
    int* rows_d = dmalloc<int>(dfl::B);
    G6RT_CK(cudaMemcpy(rows_d, rows, sizeof rows, cudaMemcpyHostToDevice));
    for (int r = 0; r < NR; ++r)
    {
        const std::string b = dir + "/dflash.r" + std::to_string(r), h = dir + "/head.r" + std::to_string(r);
        load_pack(dp[r], b + ".manifest.json", b + ".pack", r);
        load_pack(hp[r], h + ".manifest.json", h + ".pack", r);
        const Find fd = pack_find(dp[r], store), fh = pack_find(hp[r], store);
        Model& m = M[r];
        m.rank = r;
        m.emb = emb_d;
        m.head_t = bind::head_tensors([&](const char* nm) { return as_bind(fh(nm)); });
        if (!bind::vocab_slice(r, m.vlo, m.vn)) die("vocab slice");
        g6::FabricDev f{};
        f.send = dmalloc<char>(2 * (size_t)dfl::NC * stride);
        f.recv = recv; f.doorbell = bell; f.error = err;
        f.rank = r; f.nr = NR; f.ncoll = dfl::NC; f.stride = stride;
        loop.send[r] = f.send;
        out[r] = dmalloc<DflOut>(1);
        D[r].in = in; D[r].out = out[r]; D[r].nc = 0;
        const DflMeta meta = dfl_read_meta(fd);
        dfl_build(D[r], m, fd, meta, f, stepp, sel == 1, st, fill);
        D[r].emb_rows = rows_d;       // block row r reads row r of the case's table
    }
    fprintf(stderr, "# device tensors: %.2f GB held, %.2f GB shared between ranks\n", store.bytes / 1e9, store.shared_bytes / 1e9);
    loop.recv = recv; loop.stepp = stepp; loop.ncoll = dfl::NC; loop.stride = stride;
    u64 step = 0;
    const int nch = (n + dfl::B - 1) / dfl::B;
    if (fill)                         // the prompt-context taps: row i (position p0 + i) on rank i % NR
    {
        if (n > dfl::WIN) die("--fill: at most WIN context rows");
#if G6_MODEL_FLASH
        struct { float* resid4; int* pos; g6::FabricDev f; } P{dmalloc<float>((size_t)n * 4 * HID), dmalloc<int>(n), {}};
        std::vector<int> pos(n);
        for (int i = 0; i < n; ++i) pos[i] = p0 + i;
        G6RT_CK(cudaMemcpy(P.pos, pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        const bool sp = knob::glm53f_pf_sp();
        for (int r = 0; r < NR; ++r)
        {
            D[r].pf_lo = p0; D[r].pf_hi = p0 + n;
            G6RT_CK(cudaMemsetAsync(D[r].pf_feat, 0, (size_t)n * dfl::NT * HID * 2, st));
            dfl_of(M[r]) = &D[r];
            M[r].layers.resize(1);
            P.f.rank = r; P.f.nr = NR;
            const int j0 = sp ? pf_r0(n, r, NR) : 0, j1 = sp ? j0 + pf_rows(n, r, NR) : n;
            for (int s = 0; s < dfl::NT; ++s)
            {
                std::vector<float> v((size_t)n * 4 * HID, NAN);
                for (int j = j0; j < j1; ++j)
                    for (int c = 0; c < HID; ++c)   // streams x, x, 2x, 0: an exact mean, and a wrong stream shows
                    {
                        const float x = feat[((size_t)j * dfl::NT + s) * HID + c];
                        float* w = v.data() + (size_t)j * 4 * HID + c;
                        w[0] = w[HID] = x; w[2 * HID] = 2 * x; w[3 * HID] = 0.f;
                    }
                G6RT_CK(cudaMemcpy(P.resid4, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
                M[r].layers[0].l = D[r].tlayer[s];
                dfl_pf_tap_flash(M[r], P, 0, n, st);
                G6RT_CK(cudaStreamSynchronize(st));
            }
        }
#else
        for (int r = 0; r < NR; ++r)
        {
            std::vector<__half> pf((size_t)n * dfl::NT * HID, __float2half(0.f));
            for (int i = r; i < n; i += NR)
                for (int k = 0; k < dfl::NT * HID; ++k) pf[(size_t)i * dfl::NT * HID + k] = __float2half(feat[(size_t)i * dfl::NT * HID + k] * dfl::FEAT_SCALE);
            G6RT_CK(cudaMemcpy(D[r].pf_feat, pf.data(), pf.size() * 2, cudaMemcpyHostToDevice));
            D[r].pf_lo = p0; D[r].pf_hi = p0 + n;
        }
#endif
    }
    for (int c = 0; c < nch; ++c)
    {
        const int rows_c = std::min(dfl::B, n - c * dfl::B);
        const bool query = c == nch - 1;
        if (fill && !query)           // a fill step: phase 0 publishes the NT gathers, phase 1 collective NT
        {
            DflIn di;
            memset(&di, 0, sizeof di);
            di.step = ++step;
            di.ctx_rows = rows_c; di.ctx_pos0 = p0 + c * dfl::B; di.q0 = di.ctx_pos0 + rows_c; di.lo = p0;
            G6RT_CK(cudaMemcpy(in, &di, sizeof di, cudaMemcpyHostToDevice));
            for (int p = 0; p < DFL_FILL_PH; ++p)
            {
                for (Dfl& x : D) dfl_fill_phase(x, p, st);
                const std::vector<uint32_t> sz = dfl_fill_sizes();   // phase 0 ends with collectives 0..NT-1, phase 1 with NT
                for (int k = p == 0 ? 0 : dfl::NT; p < 2 && k < (p == 0 ? dfl::NT : dfl::NT + 1); ++k)
                {
                    DLoop a = loop;
                    a.c = k; a.n16 = (int)(sz[k] / 16);
                    k_dloop<<<NR, 128, 0, st>>>(a);
                }
            }
            G6RT_CK(cudaStreamSynchronize(st));
            G6RT_CK(cudaGetLastError());
            continue;
        }
        std::vector<__half> fh((size_t)dfl::B * dfl::NT * HID, __float2half(0.f));
        for (int r = 0; r < rows_c; ++r)
            for (int k = 0; k < dfl::NT * HID; ++k) fh[(size_t)r * dfl::NT * HID + k] = __float2half(feat[((size_t)c * dfl::B + r) * dfl::NT * HID + k] * dfl::FEAT_SCALE);
        for (Dfl& x : D) G6RT_CK(cudaMemcpy(x.feat, fh.data(), fh.size() * 2, cudaMemcpyHostToDevice));
        DflIn di;
        memset(&di, 0, sizeof di);
        di.step = ++step;
        di.ctx_rows = rows_c; di.ctx_pos0 = p0 + c * dfl::B; di.q0 = di.ctx_pos0 + rows_c; di.lo = lo; di.tok0 = bonus; di.query = query;
        for (int r = 0; r < dfl::B; ++r) di.srow[r] = glue::SampleRow{temp > 0 ? 1.f / temp : 0.f, seed, 0u, di.q0 + r, 0, 0.f};
        G6RT_CK(cudaMemcpy(in, &di, sizeof di, cudaMemcpyHostToDevice));
        for (int p = 0; p < (query ? dfl::NPH : 2); ++p)
        {
            for (Dfl& x : D) dfl_phase(x, p, st, query);
            if (dfl_delivered(p, query))
            {
                DLoop a = loop;
                a.c = p; a.n16 = (int)(dfl_payload(D[0], p) / 16);
                k_dloop<<<NR, 128, 0, st>>>(a);
            }
        }
        G6RT_CK(cudaStreamSynchronize(st));
        G6RT_CK(cudaGetLastError());
    }
    unsigned e = 0;
    G6RT_CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
    if (e) die("fabric error word " + std::to_string(e));
    std::vector<DflOut> o(NR);
    for (int r = 0; r < NR; ++r) G6RT_CK(cudaMemcpy(&o[r], out[r], sizeof(DflOut), cudaMemcpyDeviceToHost));
    for (int r = 1; r < NR; ++r)
        if (memcmp(o[r].tok + 1, o[0].tok + 1, sizeof(int) * dfl::K) || memcmp(o[r].lp + 1, o[0].lp + 1, sizeof(float) * dfl::K))
            die("ranks disagree on the drafts or their log-probabilities");
    printf("drafts");
    for (int r = 1; r < dfl::B; ++r) printf(" %d", o[0].tok[r]);
    printf("\nlp");                  // the bits, for a byte comparison between builds (TP4 vs the replicated TP6 body)
    for (int r = 1; r < dfl::B; ++r) { uint32_t b; memcpy(&b, &o[0].lp[r], 4); printf(" %08x", b); }
    printf("\n");
    // outputs
    const std::string of = dir + "/out." + std::to_string(sel) + ".bin";
    FILE* f = fopen(of.c_str(), "wb");
    fwrite(o[0].tok, 4, dfl::B, f);
    float* xf = dmalloc<float>((size_t)dfl::B * HID);
    k_h2f<<<48, 256, 0, st>>>(D[0].x, xf, dfl::B * HID);
    std::vector<float> xh((size_t)dfl::B * HID);
    G6RT_CK(cudaMemcpy(xh.data(), xf, xh.size() * 4, cudaMemcpyDeviceToHost));
    fwrite(xh.data(), 4, xh.size(), f);
    // the ring at the last chunk's context positions: [L][k, v][NKV][rows][HD], the KV heads from their ranks (REP: rank 0
    // holds all eight)
    const int last = n - (nch - 1) * dfl::B, pos0 = p0 + (nch - 1) * dfl::B;
    std::vector<__half> ring((size_t)dfl::L * dfl::NKR * dfl::RING * dfl::HD);
    std::vector<float> kv((size_t)dfl::L * 2 * dfl::NKV * last * dfl::HD);
    for (int r = 0; r < (dfl::REP ? 1 : NR); ++r)
        for (int w = 0; w < 2; ++w)
        {
            G6RT_CK(cudaMemcpy(ring.data(), w ? D[r].vring : D[r].kring, ring.size() * 2, cudaMemcpyDeviceToHost));
            for (int j = 0; j < dfl::L; ++j)
                for (int h = 0; h < dfl::NKR; ++h)
                    for (int i = 0; i < last; ++i)
                        for (int d = 0; d < dfl::HD; ++d)
                            kv[((((size_t)j * 2 + w) * dfl::NKV + r * dfl::NKR + h) * last + i) * dfl::HD + d] =
                                __half2float(ring[(((size_t)j * dfl::NKR + h) * dfl::RING + ((pos0 + i) & (dfl::RING - 1))) * dfl::HD + d]);
        }
    fwrite(kv.data(), 4, kv.size(), f);
    fclose(f);
    fprintf(stderr, "# wrote %s (%d context rows, last chunk %d rows at %d)\n", of.c_str(), n, last, pos0);
    return 0;
}
