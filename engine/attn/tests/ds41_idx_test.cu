// DS41 Attention B kernel test: runs every kernel of engine/attn/ds41_idx.cuh on the fixtures of engine/ref/ds41/idx_fixture.py and
// compares with the torch reference (idx_ref.py). Needs a GPU that runs sm_121a (a Spark).
//   python3 engine/ref/ds41/idx_fixture.py /tmp/ds41_idx_fx
//   nvcc -std=c++17 -gencode=arch=compute_121a,code=sm_121a -I. -Iengine engine/attn/tests/ds41_idx_test.cu -o /tmp/ds41_idx_test
//   /tmp/ds41_idx_test /tmp/ds41_idx_fx
// EXACT checks (no rope, no matmul order in the path: logits, block scores, top-k, candidates, compact, restore, weights) must have 0
// mismatches. NEAR checks (rms sum order, torch vs CUDA cos/sin and matmul order) may differ in the last bf16 bit of a few elements;
// they must stay under 0.5% of the elements. Exit code = number of failed checks.
#include "../ds41_idx.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ds41i;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static std::string g_dir;
static int g_fail = 0;

static std::vector<uint8_t> rd(const char* name)
{
    std::string p = g_dir + "/" + name + ".bin";
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) exit(2);
    fclose(f);
    return v;
}
template <class T> static T* up(const std::vector<uint8_t>& v)
{
    void* d;
    CK(cudaMalloc(&d, v.size() ? v.size() : 1));
    CK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice));
    return (T*)d;
}
template <class T> static T* dalloc(size_t n)
{
    void* d;
    CK(cudaMalloc(&d, n * sizeof(T) ? n * sizeof(T) : 1));
    CK(cudaMemset(d, 0xAB, n * sizeof(T)));
    return (T*)d;
}
template <class T> static std::vector<T> down(const void* d, size_t n)
{
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static void check(const char* name, const void* got, const std::vector<uint8_t>& want, size_t elem, bool exact)
{
    const size_t n = want.size() / elem;
    const uint8_t* g = (const uint8_t*)got;
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < n; i++)
        if (memcmp(g + i * elem, want.data() + i * elem, elem)) { if (first == (size_t)-1) first = i; bad++; }
    const bool ok = exact ? bad == 0 : bad * 200 <= n;
    printf("%-28s %s  %zu / %zu mismatch%s", name, ok ? "PASS" : "FAIL", bad, n, exact ? " (exact)" : " (near)");
    if (bad) printf("  first at %zu", first);
    printf("\n");
    if (!ok) g_fail++;
}
// a pool of pages with a scrambled page table: logical page j -> physical (j * 37) % npages (npages prime, not 37)
struct Pool
{
    uint8_t* mem;
    int32_t* dpages;
    std::vector<int32_t> pages;
    int npages;
    size_t page_bytes;
};
static Pool make_pool(int logical_pages, size_t page_bytes)
{
    Pool p;
    p.npages = 701;
    p.page_bytes = page_bytes;
    p.pages.resize(logical_pages);
    for (int j = 0; j < logical_pages; j++) p.pages[j] = (j * 37) % p.npages;
    CK(cudaMalloc(&p.mem, (size_t)p.npages * page_bytes));
    CK(cudaMemset(p.mem, 0, (size_t)p.npages * page_bytes));
    CK(cudaMalloc(&p.dpages, logical_pages * 4));
    CK(cudaMemcpy(p.dpages, p.pages.data(), logical_pages * 4, cudaMemcpyHostToDevice));
    return p;
}
// host view of a plane: gather `rows` rows of `row_bytes` from the pool into a dense array
static std::vector<uint8_t> gather(const Pool& p, int rows_per_page, size_t plane_off, int row_bytes, int row0, int rows)
{
    std::vector<uint8_t> all((size_t)p.npages * p.page_bytes);
    CK(cudaMemcpy(all.data(), p.mem, all.size(), cudaMemcpyDeviceToHost));
    std::vector<uint8_t> out((size_t)rows * row_bytes);
    for (int i = 0; i < rows; i++)
    {
        const int j = row0 + i;
        const uint8_t* src = all.data() + (size_t)p.pages[j / rows_per_page] * p.page_bytes + plane_off + (size_t)(j % rows_per_page) * row_bytes;
        memcpy(out.data() + (size_t)i * row_bytes, src, row_bytes);
    }
    return out;
}
static void scatter(const Pool& p, int rows_per_page, size_t plane_off, int row_bytes, int rows, const std::vector<uint8_t>& dense)
{
    std::vector<uint8_t> all((size_t)p.npages * p.page_bytes);
    CK(cudaMemcpy(all.data(), p.mem, all.size(), cudaMemcpyDeviceToHost));
    for (int j = 0; j < rows; j++)
        memcpy(all.data() + (size_t)p.pages[j / rows_per_page] * p.page_bytes + plane_off + (size_t)(j % rows_per_page) * row_bytes,
               dense.data() + (size_t)j * row_bytes, row_bytes);
    CK(cudaMemcpy(p.mem, all.data(), all.size(), cudaMemcpyHostToDevice));
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: ds41_idx_test <fixture dir>\n"); return 2; }
    g_dir = argv[1];
    int T, POS0, N, R;
    {
        FILE* f = fopen((g_dir + "/dims.txt").c_str(), "r");
        if (!f || fscanf(f, "%d %d %d %d", &T, &POS0, &N, &R) != 4) { fprintf(stderr, "dims.txt\n"); return 2; }
        fclose(f);
    }
    const RopeInv inv = rope_inv_ds41();
    cudaStream_t s = 0;

    // ---------------------------------------------------------------- ca row ratio 1 (rows POS0 .. POS0 + T - 1, ratio-1 page = 64 rows x 288 B)
    {
        const int lp = (POS0 + T + 63) / 64;
        Pool ca = make_pool(lp, 64 * CA_ROW);
        CaR1P p{};
        p.y = up<bf16>(rd("ca_y")); p.norm_w = up<bf16>(rd("ca_normw")); p.inv = inv; p.pos0 = POS0;
        p.ca = ca_map(ca.mem, ca.dpages, 64, ca.page_bytes);
        bf16* pre = dalloc<bf16>((size_t)T * D);
        bf16* roped = dalloc<bf16>((size_t)T * D);
        p.pre = pre; p.roped = roped;
        launch_ca_row_r1(p, T, s);
        CK(cudaDeviceSynchronize());
        check("ca pre (rmsnorm)", down<uint8_t>(pre, (size_t)T * D * 2).data(), rd("ca_pre"), 2, false);
        check("ca roped", down<uint8_t>(roped, (size_t)T * D * 2).data(), rd("ca_roped"), 2, false);
        check("ca row 288 B", gather(ca, 64, 0, CA_ROW, POS0, T).data(), rd("ca_rec"), 1, false);
        // the dequantizer reads back what the producer wrote
        bf16* dq = dalloc<bf16>((size_t)T * D);
        launch_ca_dequant(p.ca, POS0, T, dq, s);
        CK(cudaDeviceSynchronize());
        const std::vector<bf16> h = down<bf16>(dq, (size_t)T * D);
        const std::vector<uint8_t> rec = rd("ca_rec");
        size_t bad = 0;
        for (int t = 0; t < T; t++)
            for (int c = 0; c < D; c++)
            {
                const uint32_t code = (rec[(size_t)t * CA_ROW + (c >> 1)] >> ((c & 1) * 4)) & 0xF;
                static const float mag[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
                float v = mag[code & 7] * ((code & 8) && (code & 7) ? -1.f : 1.f);
                __nv_fp8_e4m3 e; e.__x = rec[(size_t)t * CA_ROW + CA_CODES + (c >> 4)];
                v *= (float)e;
                if (__bfloat162float(h[(size_t)t * D + c]) != v) bad++;
            }
        printf("%-28s %s  %zu / %d mismatch\n", "ca dequant", bad ? "FAIL" : "PASS", bad, T * D);
        if (bad) g_fail++;
    }

    // ---------------------------------------------------------------- ica row (index key) from the fixture's pre-rope latent
    {
        const int lp = (POS0 + T + 63) / 64;
        Pool ic = make_pool(lp, 64 * ICA_ROW);
        IcaP p{};
        p.pre = up<bf16>(rd("ca_pre")); p.wk = up<bf16>(rd("ica_wk")); p.knorm_w = up<bf16>(rd("ica_knormw")); p.inv = inv;
        p.row0 = POS0; p.ratio = 1;
        p.ica = ica_map(ic.mem, ic.dpages, 64, ic.page_bytes);
        bf16* kd = dalloc<bf16>((size_t)T * ID);
        p.kdbg = kd;
        launch_ica_row(p, T, s);
        CK(cudaDeviceSynchronize());
        check("ica k (norm+rope)", down<uint8_t>(kd, (size_t)T * ID * 2).data(), rd("ica_k"), 2, false);
        check("ica codes 64 B", gather(ic, 64, 0, IK_CODES, POS0, T).data(), rd("ica_pk"), 1, false);
        check("ica e8m0 4 B", gather(ic, 64, (size_t)64 * IK_CODES, IK_SF, POS0, T).data(), rd("ica_ex"), 1, false);
    }

    // ---------------------------------------------------------------- index q + weights
    const bf16* qb = up<bf16>(rd("q_b"));
    const bf16* wp = up<bf16>(rd("q_wproj"));
    uint8_t* qpk = dalloc<uint8_t>((size_t)R * IH * IK_CODES);
    uint8_t* qex = dalloc<uint8_t>((size_t)R * IH * IK_SF);
    bf16* qw = dalloc<bf16>((size_t)R * IH);
    {
        IdxQP p{qb, wp, inv, POS0, qpk, qex, qw};
        launch_index_q(p, R, s);
        CK(cudaDeviceSynchronize());
        check("index q codes", down<uint8_t>(qpk, (size_t)R * IH * IK_CODES).data(), rd("q_pk"), 1, false);
        check("index q e8m0", down<uint8_t>(qex, (size_t)R * IH * IK_SF).data(), rd("q_ex"), 1, false);
        check("index weights x 2^-6", down<uint8_t>(qw, (size_t)R * IH * 2).data(), rd("q_w"), 2, true);
    }

    // ---------------------------------------------------------------- logits over N paged keys (fixture q, so no rope in the path)
    const int ldp = (int)Scratch::ld_of(N);
    Pool kp = make_pool((N + 63) / 64, 64 * ICA_ROW);
    scatter(kp, 64, 0, IK_CODES, N, rd("k_pk"));
    scatter(kp, 64, (size_t)64 * IK_CODES, IK_SF, N, rd("k_ex"));
    const IcaMap kmap = ica_map(kp.mem, kp.dpages, 64, kp.page_bytes);
    const int32_t* klen = up<int32_t>(rd("klen"));
    Scratch sc;
    {
        void* mem;
        CK(cudaMalloc(&mem, Scratch::bytes(R, N)));
        CK(cudaMemset(mem, 0xAB, Scratch::bytes(R, N)));
        sc = Scratch::carve(mem, R, N);
    }
    const uint8_t* fq_pk = up<uint8_t>(rd("q_pk"));
    const uint8_t* fq_ex = up<uint8_t>(rd("q_ex"));
    const bf16* fq_w = up<bf16>(rd("q_w"));
    {
        LogitsP l{};
        l.q_pack = fq_pk; l.q_exp = fq_ex; l.w = fq_w; l.k = kmap; l.klen = klen; l.p = sc.p; l.ldp = sc.ldp; l.ncols = N; l.R = R;
        for (int mma = 0; mma < 2; mma++)
        {
            CK(cudaMemset(sc.p, 0xAB, (size_t)R * ldp * 2));
            if (mma) launch_logits_dense_mma(l, s);
            else launch_logits_dense(l, s);
            CK(cudaDeviceSynchronize());
            const std::vector<uint8_t> want = rd("p");
            const std::vector<uint8_t> got = down<uint8_t>(sc.p, (size_t)R * ldp * 2);
            std::vector<uint8_t> dense((size_t)R * N * 2);
            for (int r = 0; r < R; r++) memcpy(dense.data() + (size_t)r * N * 2, got.data() + (size_t)r * ldp * 2, (size_t)N * 2);
            check(mma ? "logits dense (tensor-core path)" : "logits dense", dense.data(), want, 2, true);
        }
    }
    // ---------------------------------------------------------------- top-512 on the fixture logits
    {
        std::vector<uint8_t> pw = rd("p");
        std::vector<uint8_t> padded((size_t)R * ldp * 2, 0);
        for (int r = 0; r < R; r++) memcpy(padded.data() + (size_t)r * ldp * 2, pw.data() + (size_t)r * N * 2, (size_t)N * 2);
        CK(cudaMemcpy(sc.p, padded.data(), padded.size(), cudaMemcpyHostToDevice));
        int32_t* out = dalloc<int32_t>((size_t)R * TOPK);
        launch_topk<TOPK>(sc.p, sc.ldp, klen, ldp, out, TOPK, R, s);
        CK(cudaDeviceSynchronize());
        check("topk 512 ascending", down<uint8_t>(out, (size_t)R * TOPK * 4).data(), rd("topk"), 4, true);
        // the candidate stage
        IndexIn in{};
        in.R = R; in.klen = klen; in.max_klen = N;
        stage_cand_blocks(in, sc, s);
        CK(cudaDeviceSynchronize());
        const std::vector<uint8_t> bs = rd("bscore");
        const int nbv = N / 8;
        const std::vector<uint8_t> gs = down<uint8_t>(sc.sc, (size_t)R * sc.ldsc * 2);
        std::vector<uint8_t> cmp((size_t)R * nbv * 2);
        for (int r = 0; r < R; r++) memcpy(cmp.data() + (size_t)r * nbv * 2, gs.data() + (size_t)r * sc.ldsc * 2, (size_t)nbv * 2);
        check("block scores", cmp.data(), bs, 2, true);
        check("valid blocks", down<uint8_t>(sc.nvalid, R * 4).data(), rd("nvalid"), 4, true);
        check("candidate blocks 2048", down<uint8_t>(sc.cand, (size_t)R * CAND_BLOCKS * 4).data(), rd("cand"), 4, true);
    }
    // ---------------------------------------------------------------- candidate consumer (second layer's q)
    {
        const int32_t* cand = up<int32_t>(rd("cand"));
        const uint8_t* q2pk = up<uint8_t>(rd("q2_pk"));
        const uint8_t* q2ex = up<uint8_t>(rd("q2_ex"));
        const bf16* q2w = up<bf16>(rd("q2_w"));
        LogitsP l{};
        l.q_pack = q2pk; l.q_exp = q2ex; l.w = q2w; l.k = kmap; l.klen = klen; l.cand = cand; l.ldc = CAND_BLOCKS;
        l.p = sc.cp; l.ldp = CAND_COLS; l.ncols = CAND_COLS; l.R = R;
        launch_logits_gather(l, s);
        launch_compact_lens(cand, CAND_BLOCKS, klen, sc.clen, R, s);
        launch_topk<TOPK>(sc.cp, CAND_COLS, sc.clen, CAND_COLS, sc.ci, TOPK, R, s);
        int32_t* out = dalloc<int32_t>((size_t)R * TOPK);
        launch_restore(sc.ci, TOPK, cand, CAND_BLOCKS, out, TOPK, R, s);
        CK(cudaDeviceSynchronize());
        check("logits gather (compact)", down<uint8_t>(sc.cp, (size_t)R * CAND_COLS * 2).data(), rd("cp"), 2, true);
        check("compact lens", down<uint8_t>(sc.clen, R * 4).data(), rd("clen"), 4, true);
        check("compact top-512", down<uint8_t>(sc.ci, (size_t)R * TOPK * 4).data(), rd("ci"), 4, true);
        check("restored top-512", down<uint8_t>(out, (size_t)R * TOPK * 4).data(), rd("cons_topk"), 4, true);
    }
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail;
}
