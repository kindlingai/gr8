// ds41_attn_test --mgate: per-op unit gate "a chunk of M rows at start > 0 == M single-row calls" (DS-gr8 prefill design, step 2).
// Included by ds41_attn_test.cu INSIDE namespace ds41a (it uses that file's CK, dalloc, dcopy, cc_fill, bf2f, make_records and gv hooks).
// No fixture and no checkpoint: synthetic fp8 / bf16 weights of the real shapes (16 heads = one TP4 rank), deterministic inputs.
//
//   ds41_attn_test --mgate [--M 16,32] [--start 4,5,156,157,300,301] [--kind r2,r1,w] [--gv 0|1|both] [--tk N] [--reps N] [--ops proj,comp,core,out,layer]
//   --cpa: after each core line also run the core with k_attn_cpa (GR8_DS_ATTN_CPA=1) over the same inputs: a 'core_cpa' line, bit-compared to the same single-row k_attn reference, timed the same way.
//   --ops: which op blocks are compared and timed (default all; 'core' also runs its canary-leak control). The start state and the single-row reference are always built.
//
// One line per (op, M, start, kind, gv):   op M start us bitexact kind=.. gv=.. ok=0|1 seq_us=.. [detail]
//   us        median of --reps (7) event-timed runs of the CHUNK op at M rows (device time, launch gaps of the op included)
//   seq_us    the same op summed over the M single-row calls of the reference (one event pair per call)
//   bitexact  yes = every bit equal | class = not equal, all finite, rrms < 2e-2 (a rounding-class difference: GEMV summation order
//             flips an fp8 requant step) | no = anything else
//   ok        1 = the line meets its expectation: yes everywhere at gv=0 and for comp / core / canary-leak at gv=1; class is accepted only
//             at gv=1, M > 8, ops proj / out / layer (the GEMV serves <= 8 rows, the generic GEMM above: a different fp32 order, design item 2)
// Ops (each teacher-forced: the chunk op is fed the single-row reference's own input, so one op's difference does not hide behind another's):
//   proj     attn_proj chunk from y -> S.q, S.kvrows, the whole window ring == the M single-row calls
//   comp     (r2) comp_row chunk -> the whole C.comp (poisoned rows past the last written one included), S.cnorm completions, the live
//            pending half in C.kvs / C.scs (only when the last position is even; the slot is dead otherwise). Odd start = the chunk
//            opens with the pending half of the previous call; odd end = it leaves one.
//   core     attn_core chunk over the reference's FINAL ring + C.comp, with the reference's q -> S.o. The final C.comp holds every row
//            the chunk completes, i.e. the FUTURE rows of the earlier chunk rows are physically present (fut=), while the reference
//            call of row t saw none: the future-row canary for the list-driven layers. Rows past the last written one are NaN.
//   canary-leak  control: the same call with a list that also holds the first future row of every row (j = klen). It must differ
//            (bitexact no, ok 1): if it did not, the core test could not see a leaked future row.
//   out      attn_out chunk from the reference's S.o -> out
//   layer    layer_forward chunk from y -> out (everything chained; at gv=1 the proj difference propagates)
// Kinds: r2 = ratio-2 KV source (L2/8/14: comp_row writes C.comp, list klen = (p+1)/2); r1 = ratio-1 reader (L20..39: C.comp rows come from
// the indexer, here a master array written one row per position before that row's call; klen = p+1); w = window only (L0, L1, DSpark layers).
// The list is the indexer's contract: ascending, -1 tail, every row j < klen(p) = (p+1)/ratio; klen > tk keeps tk evenly spaced rows.
// The start state is built by single-row calls from position 0 (snapshots at each start), so the reference is the engine's own decode.
// Exit 0 when every line has ok=1.
namespace mg
{
constexpr int HEADS = 16;
static uint64_t g_s = 0x1234567ull;
static uint32_t rnd() { g_s ^= g_s << 13; g_s ^= g_s >> 7; g_s ^= g_s << 17; return (uint32_t)(g_s >> 16); }
static std::vector<uint8_t> codes(size_t n, uint64_t seed)   // finite E4M3 codes, |v| <= 3.75
{
    g_s = seed * 0x9E3779B97F4A7C15ull + 1;
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) { const uint32_t r = rnd(); v[i] = (uint8_t)(((r & 1) << 7) | (((r >> 1) % 9) << 3) | ((r >> 8) & 7)); }
    return v;
}
static std::vector<uint8_t> scales(size_t n, uint64_t seed, int lo)   // E8M0 bytes lo .. lo + 2
{
    g_s = seed * 0x9E3779B97F4A7C15ull + 7;
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint8_t)(lo + rnd() % 3);
    return v;
}
// rows_alloc rows of codes (a grouped matrix allocates groups * N), N / K = what one call sees
static Fp8Mat mat(long rows_alloc, long N, long K, uint64_t seed, int slo)
{
    Fp8Mat m{};
    m.w = dcopy(codes((size_t)rows_alloc * K, seed));
    m.s = dcopy(scales((size_t)rows_alloc * (K / 32), seed + 1, slo));
    m.N = (int)N;
    m.K = (int)K;
    return m;
}
static std::vector<uint16_t> bf_vec(size_t n, uint64_t seed, float lo, float hi)
{
    std::vector<uint16_t> v(n);
    cc_fill(v, 0, (int)n, seed, 1.0f);
    for (size_t i = 0; i < n; i++) { const float u = 0.5f * (bf2f(v[i]) + 1.0f); v[i] = cc_f2bf(lo + (hi - lo) * u); }
    return v;
}

struct Kind { const char* name; int ratio; bool comp; };   // ratio 0 = window only
static const Kind KINDS[3] = {{"r2", 2, true}, {"r1", 1, false}, {"w", 0, false}};

// ascending list of the visible rows, -1 tail; klen > tk keeps tk evenly spaced rows. leak: the first row past the visible ones is added.
static void make_list(int* o, int tk, int klen, bool leak)
{
    for (int i = 0; i < tk; i++) o[i] = klen <= tk ? (i < klen ? i : -1) : (int)((long)i * klen / tk);
    if (leak) o[klen < tk ? klen : tk - 1] = klen;
}

struct Cmp { long n = 0, ndiff = 0; bool nonfinite = false; double rrms = 0; };
static Cmp cmp_bf16(const void* dg, const void* de, long n)
{
    std::vector<uint16_t> g((size_t)n), e((size_t)n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(g.data(), dg, (size_t)n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(e.data(), de, (size_t)n * 2, cudaMemcpyDeviceToHost));
    Cmp c;
    c.n = n;
    double d2 = 0, t2 = 0;
    for (long i = 0; i < n; i++)
    {
        if (g[i] == e[i]) continue;
        c.ndiff++;
        const double x = bf2f(g[i]), y = bf2f(e[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) { c.nonfinite = true; continue; }
        d2 += (x - y) * (x - y);
        t2 += y * y;
    }
    c.rrms = t2 > 0 ? sqrt(d2 / t2) : (d2 > 0 ? 1e9 : 0.0);
    return c;
}
static Cmp cmp_u32(const void* dg, const void* de, long n)   // fp32 state words, bitwise
{
    std::vector<uint32_t> g((size_t)n), e((size_t)n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(g.data(), dg, (size_t)n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(e.data(), de, (size_t)n * 4, cudaMemcpyDeviceToHost));
    Cmp c;
    c.n = n;
    for (long i = 0; i < n; i++) c.ndiff += g[i] != e[i];
    c.rrms = c.ndiff ? 1.0 : 0.0;
    return c;
}
// GR8_TEST_DS41_MGATE_DIAG=1: for a kv compare that differs, print every differing element (up to 8): position, column, the chunk's / the single-row call's
// value after kv prep, and the same element of the wkv output (kv0) before kv prep, plus how many kv0 elements differ in all.
static void diag_kv(const char* tag, const bf16* dkv, const bf16* dkv_seq, const bf16* dkv0, const bf16* dkv0_seq, long n, int width, int s, int M)
{
    std::vector<uint16_t> a((size_t)n), b((size_t)n), c((size_t)n), d((size_t)n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(a.data(), dkv, (size_t)n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), dkv_seq, (size_t)n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.data(), dkv0, (size_t)n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(d.data(), dkv0_seq, (size_t)n * 2, cudaMemcpyDeviceToHost));
    long n0 = 0, nk = 0;
    for (long i = 0; i < n; i++) { n0 += c[i] != d[i]; nk += a[i] != b[i]; }
    if (!n0 && !nk) return;
    printf("# diag %s M=%d start=%d: kv0 (wkv out) differs in %ld of %ld elements, kv (after prep) in %ld\n", tag, M, s, n0, n, nk);
    int shown = 0;
    for (long i = 0; i < n && shown < 8; i++)
        if (a[i] != b[i])
        {
            shown++;
            const double g = bf2f(a[i]), e = bf2f(b[i]);
            printf("#   kv diff: pos %ld col %ld (%s) chunk %.6g single %.6g rel %.4f | kv0 chunk %.6g (0x%04x) single %.6g (0x%04x)\n", s + i / width, i % width,
                   i % width >= 448 ? "rope" : "nope", g, e, e != 0 ? fabs(g - e) / fabs(e) : 0.0, bf2f(c[i]), c[i], bf2f(d[i]), d[i]);
        }
}
// rows of heads*512 that differ at all
static long rows_differ(const void* dg, const void* de, int rows, long width)
{
    std::vector<uint16_t> g((size_t)rows * width), e((size_t)rows * width);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(g.data(), dg, g.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(e.data(), de, e.size() * 2, cudaMemcpyDeviceToHost));
    long nd = 0;
    for (int r = 0; r < rows; r++) nd += memcmp(&g[(size_t)r * width], &e[(size_t)r * width], (size_t)width * 2) != 0;
    return nd;
}
struct Verdict
{
    int cls = 0;   // 0 yes, 1 class, 2 no
    std::string detail;
    double rrms = 0;
    void add(const char* nm, const Cmp& c)
    {
        const int k = c.ndiff == 0 ? 0 : (!c.nonfinite && c.rrms < 2e-2 ? 1 : 2);
        cls = std::max(cls, k);
        rrms = std::max(rrms, c.rrms);
        if (k)
        {
            char b[96];
            snprintf(b, sizeof b, " %s=%s(%ld/%ld,rrms=%.1e)", nm, k == 1 ? "class" : "no", c.ndiff, c.n, c.rrms);
            detail += b;
        }
    }
};

static double median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2] * 1000.0; }
template <class F> static double tus(F fn, int reps)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    fn();   // warm
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < reps; i++)
    {
        cudaEventRecord(a);
        fn();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float t;
        cudaEventElapsedTime(&t, a, b);
        ms.push_back(t);
    }
    CK(cudaGetLastError());
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return median(ms);
}

struct Snap { bf16 *ring, *comp; float *kvs, *scs; };
static std::vector<int> parse_ints(const char* s)
{
    std::vector<int> v;
    for (const char* p = s; *p;)
    {
        v.push_back(atoi(p));
        const char* c = strchr(p, ',');
        if (!c) break;
        p = c + 1;
    }
    return v;
}
}   // namespace mg

static int mgate_test(int argc, char** argv)
{
    using namespace mg;
    std::vector<int> Ms = {16, 32}, starts, kinds = {0, 1, 2};
    int gvmode = 2, TK = 2048, reps = 7;
    bool cpa = false;
    std::string ops = ",proj,comp,core,out,layer,";
    auto want = [&](const char* o) { return ops.find(std::string(",") + o + ",") != std::string::npos; };
    for (int i = 2; i < argc; i++)
    {
        if (!strcmp(argv[i], "--M") && i + 1 < argc) Ms = parse_ints(argv[++i]);
        else if (!strcmp(argv[i], "--start") && i + 1 < argc) starts = parse_ints(argv[++i]);
        else if (!strcmp(argv[i], "--tk") && i + 1 < argc) TK = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpa")) cpa = true;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, atoi(argv[++i]));
        else if (!strcmp(argv[i], "--ops") && i + 1 < argc) ops = std::string(",") + argv[++i] + ",";
        else if (!strcmp(argv[i], "--gv") && i + 1 < argc) { const char* a = argv[++i]; gvmode = !strcmp(a, "both") ? 2 : atoi(a); }
        else if (!strcmp(argv[i], "--kind") && i + 1 < argc)
        {
            kinds.clear();
            for (const char* p = argv[++i]; *p;)
            {
                for (int k = 0; k < 3; k++) if (!strncmp(p, KINDS[k].name, strlen(KINDS[k].name)) && (p[strlen(KINDS[k].name)] == ',' || !p[strlen(KINDS[k].name)])) kinds.push_back(k);
                const char* c = strchr(p, ',');
                if (!c) break;
                p = c + 1;
            }
        }
        else if (strcmp(argv[i], "--mgate")) { fprintf(stderr, "mgate: unknown option %s\n", argv[i]); return 2; }
    }
    int Mmax = 32;
    for (int m : Ms) Mmax = std::max(Mmax, m);
    for (int m : Ms) if (m < 1) { fprintf(stderr, "mgate: M must be >= 1\n"); return 2; }
    const int RR = 127 + Mmax;   // ring rows: 128 + MX - 1, the engine's (MX = 32 -> 159)
    if (starts.empty()) starts = {4, 5, RR - 3, RR - 2, 300, 301};
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    const int maxstart = starts.back(), CR = maxstart + Mmax + 4;   // comp rows allocated (ratio 1: one per position)
    const int G = HEADS / HPG;
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("# device %s sm_%d%d | --mgate: chunk of M rows at start vs M single-row calls | ring_rows %d heads %d tk %d reps %d | synthetic weights\n",
           pr.name, pr.major, pr.minor, RR, HEADS, TK, reps);
    printf("# columns: op M start us bitexact kind gv ok seq_us [detail]   (bitexact: yes | class = finite, rrms < 2e-2 | no)\n");
    if (Mmax > 32) printf("# NOTE M > 32 is beyond the engine's MX: ring_rows %d here\n", RR);

    // ---- weights (base set; r1 and w drop the compressor)
    Weights W0{};
    W0.wq_a = mat(QA, QA, HID, 21, 119);
    W0.wq_b = mat((long)HEADS * HD, (long)HEADS * HD, QA, 22, 123);
    W0.wkv = mat(HD, HD, HID, 23, 119);
    W0.wo_a = mat((long)G * OLORA, OLORA, GROUP_IN, 24, 118);
    W0.wo_b = mat(HID, HID, (long)G * OLORA, 25, 118);
    W0.q_norm = dbf(bf_vec(QA, 26, 0.5f, 1.5f));
    W0.kv_norm = dbf(bf_vec(HD, 27, 0.5f, 1.5f));
    W0.sink = dcopy(std::vector<float>([&] { std::vector<float> s(HEADS); for (int h = 0; h < HEADS; h++) s[h] = -1.0f + 2.0f * (float)h / (HEADS - 1); return s; }()));
    {
        std::vector<uint16_t> hk(HD * (size_t)HID), hg(HD * (size_t)HID);
        cc_fill(hk, 0, (int)hk.size(), 11, 0.03f);
        cc_fill(hg, 0, (int)hg.size(), 12, 0.03f);
        std::vector<uint16_t> hn(HD);
        for (int c = 0; c < HD; c++) hn[c] = cc_f2bf(0.5f + (float)(c % 7) * 0.25f);
        W0.c_wkv = dbf(hk);
        W0.c_wgate = dbf(hg);
        W0.c_norm = dbf(hn);
    }
    float inv_c[32], inv_w[32];
    inv_freqs_host(160000.0, 65536, inv_c);
    inv_freqs_host(10000.0, 0, inv_w);
    const float* d_inv_c = dcopy(std::vector<float>(inv_c, inv_c + 32));
    const float* d_inv_w = dcopy(std::vector<float>(inv_w, inv_w + 32));
    if (gvmode != 0)
    {
        W0.wq_a.gv = make_records(W0.wq_a.w, W0.wq_a.s, QA, HID);
        W0.wq_b.gv = make_records(W0.wq_b.w, W0.wq_b.s, (long)HEADS * HD, QA);
        W0.wkv.gv = make_records(W0.wkv.w, W0.wkv.s, HD, HID);
        W0.wo_a.gv = make_records(W0.wo_a.w, W0.wo_a.s, (long)G * OLORA, GROUP_IN);
        W0.wo_b.gv = make_records(W0.wo_b.w, W0.wo_b.s, HID, G * OLORA);
        g_fn_xfrag = gv_xfrag_launch;
        g_fn_gemv = gv_gemv_launch;
    }
    CK(cudaDeviceSynchronize());

    // ---- buffers
    cudaStream_t st = 0;
    void* sbase;
    CK(cudaMalloc(&sbase, Scratch::bytes(Mmax, HEADS)));
    CK(cudaMemset(sbase, 0, Scratch::bytes(Mmax, HEADS)));
    Scratch S;
    S.carve(sbase, Mmax, HEADS);
    const long QW = (long)HEADS * HD;
    bf16* d_y = dalloc<bf16>((size_t)Mmax * HID);       // chunk input
    bf16* d_y1 = dalloc<bf16>(HID);
    int* d_pos = dalloc<int>(1);
    int* d_tk1 = dalloc<int>(TK);
    int* d_tkM = dalloc<int>((size_t)Mmax * TK);
    int* d_tkL = dalloc<int>((size_t)Mmax * TK);
    bf16* d_out = dalloc<bf16>((size_t)Mmax * HID);
    bf16* q_seq = dalloc<bf16>((size_t)Mmax * QW);
    bf16* kv_seq = dalloc<bf16>((size_t)Mmax * HD);
    bf16* kv0_seq = dalloc<bf16>((size_t)Mmax * HD);   // MGATE_DIAG: the single-row calls' wkv output
    bf16* o_seq = dalloc<bf16>((size_t)Mmax * QW);
    bf16* out_seq = dalloc<bf16>((size_t)Mmax * HID);
    bf16* cn_seq = dalloc<bf16>((size_t)(Mmax / 2 + 2) * HD);
    Snap fin{dalloc<bf16>((size_t)RR * HD), dalloc<bf16>((size_t)CR * HD), dalloc<float>(2 * HD), dalloc<float>(2 * HD)};   // the reference's final state
    Cache C{};
    C.ring = dalloc<bf16>((size_t)RR * HD);
    C.ring_rows = RR;
    C.comp = dalloc<bf16>((size_t)CR * HD);
    C.kvs = dalloc<float>(2 * HD);
    C.scs = dalloc<float>(2 * HD);
    bf16* d_master = dalloc<bf16>((size_t)CR * HD);   // r1: the compressed row of every position, written one per position
    {
        std::vector<uint16_t> m((size_t)CR * HD);
        cc_fill(m, 0, (int)m.size(), 31, 1.5f);
        CK(cudaMemcpy(d_master, m.data(), m.size() * 2, cudaMemcpyHostToDevice));
    }
    auto ypos = [&](int p) { std::vector<uint16_t> y(HID); cc_fill(y, 0, HID, 1000 + (uint64_t)p, 1.0f); return y; };
    auto poison_comp = [&](bf16* c) { CK(cudaMemset(c, 0xFF, (size_t)CR * HD * 2)); };
    auto save = [&](Snap& s) {
        CK(cudaMemcpy(s.ring, C.ring, (size_t)RR * HD * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(s.comp, C.comp, (size_t)CR * HD * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(s.kvs, C.kvs, 2 * HD * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(s.scs, C.scs, 2 * HD * 4, cudaMemcpyDeviceToDevice));
    };
    auto load = [&](const Snap& s) {
        CK(cudaMemcpy(C.ring, s.ring, (size_t)RR * HD * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(C.comp, s.comp, (size_t)CR * HD * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(C.kvs, s.kvs, 2 * HD * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(C.scs, s.scs, 2 * HD * 4, cudaMemcpyDeviceToDevice));
    };
    auto set_pos = [&](int p) { CK(cudaMemcpy(d_pos, &p, 4, cudaMemcpyHostToDevice)); };

    int lines = 0, bad = 0;
    auto emit = [&](const char* op, int M, int start, double us, const char* bx, const Kind& K, int gv, bool ok, double seq_us, const std::string& extra) {
        printf("%-11s %3d %5d %9.1f %-5s kind=%s gv=%d ok=%d seq_us=%.1f%s\n", op, M, start, us, bx, K.name, gv, (int)ok, seq_us, extra.c_str());
        lines++;
        bad += !ok;
    };
    bool sanity_done = false;

    for (int ki : kinds)
    {
        const Kind& K = KINDS[ki];
        Weights W = W0;
        W.inv = K.ratio == 0 ? d_inv_w : d_inv_c;
        if (!K.comp) W.c_wkv = W.c_wgate = W.c_norm = nullptr;
        const int tk = K.ratio ? TK : 0;
        const bf16* comp_src = K.ratio ? C.comp : nullptr;
        const int* topk1 = K.ratio ? d_tk1 : nullptr;
        auto klen_of = [&](int p) { return K.ratio ? (p + 1) / K.ratio : 0; };
        std::vector<int> hl((size_t)TK);
        auto upload_list = [&](int p) { make_list(hl.data(), TK, klen_of(p), false); CK(cudaMemcpy(d_tk1, hl.data(), (size_t)TK * 4, cudaMemcpyHostToDevice)); };
        auto write_row = [&](int j) { CK(cudaMemcpy(C.comp + (size_t)j * HD, d_master + (size_t)j * HD, HD * 2, cudaMemcpyDeviceToDevice)); };   // r1: the indexer's row of position j
        auto step1 = [&](int p) {   // one single-row layer call at position p (the engine's decode form: start from a device word)
            CK(cudaMemcpy(d_y1, ypos(p).data(), HID * 2, cudaMemcpyHostToDevice));
            set_pos(p);
            if (K.ratio) upload_list(p);
            if (K.ratio == 1) write_row(p);
            layer_forward(W, C, comp_src, S, d_y1, 1, 0, HEADS, topk1, TK, tk, d_out, st, d_pos);
        };

        // ---- start states: single-row calls from position 0, a snapshot at every start
        g_use_gv = false;   // start states always through the generic GEMM (any valid state does; the gv loop below sets its own mode)
        CK(cudaMemset(C.ring, 0, (size_t)RR * HD * 2));
        CK(cudaMemset(C.kvs, 0, 2 * HD * 4));
        CK(cudaMemset(C.scs, 0, 2 * HD * 4));
        poison_comp(C.comp);
        std::vector<Snap> snaps(starts.size());
        for (auto& s : snaps) s = Snap{dalloc<bf16>((size_t)RR * HD), dalloc<bf16>((size_t)CR * HD), dalloc<float>(2 * HD), dalloc<float>(2 * HD)};
        {
            size_t si = 0;
            for (int p = 0; p < maxstart; p++)
            {
                step1(p);
                if (si < starts.size() && starts[si] == p + 1) save(snaps[si++]);
            }
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
        }

        for (size_t si = 0; si < starts.size(); si++)
            for (int M : Ms)
                for (int gv = 0; gv < 2; gv++)
                {
                    if ((gv == 0 && gvmode == 1) || (gv == 1 && gvmode == 0)) continue;
                    const int s = starts[si];
                    g_use_gv = gv != 0;
                    const bool lenient = gv && M > 8;   // GEMV (<= 8 rows) vs generic GEMM: class accepted for proj / out / layer
                    std::string ex;

                    // ---- the reference: M single-row calls from the snapshot, one event pair per op
                    load(snaps[si]);
                    std::vector<cudaEvent_t> ev((size_t)M * 10 + 2);
                    for (auto& e : ev) cudaEventCreate(&e);
                    CK(cudaDeviceSynchronize());
                    int ncn = 0;
                    for (int t = 0; t < M; t++)
                    {
                        const int p = s + t;
                        CK(cudaMemcpy(d_y1, ypos(p).data(), HID * 2, cudaMemcpyHostToDevice));
                        set_pos(p);
                        if (K.ratio) upload_list(p);
                        if (K.ratio == 1) write_row(p);
                        cudaEvent_t* e = &ev[(size_t)t * 10];
                        cudaEventRecord(e[0]);
                        attn_proj(W, C, S, d_y1, 1, 0, HEADS, st, d_pos);
                        cudaEventRecord(e[1]);
                        if (K.comp) comp_row(W, C, S, d_y1, 1, 0, st, d_pos);
                        cudaEventRecord(e[2]);
                        attn_core(W, C, comp_src, S, 1, 0, HEADS, topk1, TK, tk, st, d_pos);
                        cudaEventRecord(e[3]);
                        attn_out(W, S, 1, 0, HEADS, out_seq + (size_t)t * HID, st, d_pos);
                        cudaEventRecord(e[4]);
                        CK(cudaMemcpyAsync(q_seq + (size_t)t * QW, S.q, QW * 2, cudaMemcpyDeviceToDevice, st));
                        CK(cudaMemcpyAsync(kv_seq + (size_t)t * HD, S.kvrows, HD * 2, cudaMemcpyDeviceToDevice, st));
                        CK(cudaMemcpyAsync(kv0_seq + (size_t)t * HD, S.kv0, HD * 2, cudaMemcpyDeviceToDevice, st));
                        CK(cudaMemcpyAsync(o_seq + (size_t)t * QW, S.o, QW * 2, cudaMemcpyDeviceToDevice, st));
                        if (K.comp && (p & 1)) CK(cudaMemcpyAsync(cn_seq + (size_t)(ncn++) * HD, S.cnorm, HD * 2, cudaMemcpyDeviceToDevice, st));
                    }
                    CK(cudaDeviceSynchronize());
                    CK(cudaGetLastError());
                    double sq_proj = 0, sq_comp = 0, sq_core = 0, sq_out = 0;
                    for (int t = 0; t < M; t++)
                    {
                        cudaEvent_t* e = &ev[(size_t)t * 10];
                        float a;
                        cudaEventElapsedTime(&a, e[0], e[1]); sq_proj += a;
                        cudaEventElapsedTime(&a, e[1], e[2]); sq_comp += a;
                        cudaEventElapsedTime(&a, e[2], e[3]); sq_core += a;
                        cudaEventElapsedTime(&a, e[3], e[4]); sq_out += a;
                    }
                    for (auto& e : ev) cudaEventDestroy(e);
                    sq_proj *= 1000; sq_comp *= 1000; sq_core *= 1000; sq_out *= 1000;
                    const double sq_layer = sq_proj + sq_comp + sq_core + sq_out;
                    save(fin);
                    const int Rfinal = K.ratio ? (s + M) / K.ratio : 0;   // rows written in the reference's final C.comp
                    if (!sanity_done)
                    {
                        std::vector<uint16_t> h((size_t)M * HID);
                        CK(cudaMemcpy(h.data(), out_seq, h.size() * 2, cudaMemcpyDeviceToHost));
                        double r = 0, mx = 0;
                        long nf = 0;
                        for (uint16_t b : h) { const double v = bf2f(b); if (!std::isfinite(v)) { nf++; continue; } r += v * v; mx = std::max(mx, fabs(v)); }
                        printf("# sanity (first case): reference out rms %.3e maxabs %.3e nonfinite %ld (all-zero or non-finite output would make the gate vacuous)\n", sqrt(r / h.size()), mx, nf);
                        if (nf || mx == 0) { printf("# FAIL sanity\n"); bad++; }
                        sanity_done = true;
                    }

                    // chunk input
                    {
                        std::vector<uint16_t> y((size_t)M * HID);
                        for (int t = 0; t < M; t++) { auto r = ypos(s + t); memcpy(&y[(size_t)t * HID], r.data(), HID * 2); }
                        CK(cudaMemcpy(d_y, y.data(), y.size() * 2, cudaMemcpyHostToDevice));
                    }
                    set_pos(s);
                    std::vector<int> lh((size_t)M * TK), ll((size_t)M * TK);
                    long fut_total = 0, fut_max = 0;
                    if (K.ratio)
                    {
                        for (int t = 0; t < M; t++)
                        {
                            make_list(&lh[(size_t)t * TK], TK, klen_of(s + t), false);
                            make_list(&ll[(size_t)t * TK], TK, klen_of(s + t), true);
                            const long fut = std::max(0, Rfinal - klen_of(s + t));
                            fut_total += fut;
                            fut_max = std::max(fut_max, fut);
                        }
                        CK(cudaMemcpy(d_tkM, lh.data(), lh.size() * 4, cudaMemcpyHostToDevice));
                        CK(cudaMemcpy(d_tkL, ll.data(), ll.size() * 4, cudaMemcpyHostToDevice));
                    }

                    // ---- proj
                    if (want("proj"))
                    {
                        load(snaps[si]);
                        auto fn = [&] { attn_proj(W, C, S, d_y, M, 0, HEADS, st, d_pos); };
                        fn();
                        Verdict v;
                        v.add("q", cmp_bf16(S.q, q_seq, (long)M * QW));
                        v.add("kv", cmp_bf16(S.kvrows, kv_seq, (long)M * HD));
                        if (mgate_diag()) diag_kv(K.name, S.kvrows, kv_seq, S.kv0, kv0_seq, (long)M * HD, HD, s, M);
                        v.add("ring", cmp_bf16(C.ring, fin.ring, (long)RR * HD));
                        const double us = tus(fn, reps);
                        emit("proj", M, s, us, v.cls == 0 ? "yes" : v.cls == 1 ? "class" : "no", K, gv, v.cls == 0 || (v.cls == 1 && lenient), sq_proj, v.detail);
                    }
                    // ---- comp (r2): odd start = the chunk opens on the previous call's pending half, odd end leaves one
                    if (K.comp && want("comp"))
                    {
                        load(snaps[si]);
                        auto fn = [&] { comp_row(W, C, S, d_y, M, 0, st, d_pos); };
                        fn();
                        Verdict v;
                        v.add("comp", cmp_bf16(C.comp, fin.comp, (long)CR * HD));
                        v.add("cnorm", cmp_bf16(S.cnorm, cn_seq, (long)ncn * HD));
                        const bool live = ((s + M - 1) & 1) == 0;
                        if (live)
                        {
                            v.add("kvs0", cmp_u32(C.kvs, fin.kvs, HD));
                            v.add("scs0", cmp_u32(C.scs, fin.scs, HD));
                        }
                        char b[64];
                        snprintf(b, sizeof b, " start=%s end=%s pending=%s", (s & 1) ? "odd" : "even", ((s + M - 1) & 1) ? "odd" : "even", live ? "checked" : "dead");
                        const double us = tus(fn, reps);
                        emit("comp", M, s, us, v.cls == 0 ? "yes" : v.cls == 1 ? "class" : "no", K, gv, v.cls == 0, sq_comp, v.detail + b);
                    }
                    // ---- core, teacher-forced on the reference's final ring / C.comp / q: future rows are present for the earlier rows
                    if (want("core"))
                    {
                        CK(cudaMemcpy(C.ring, fin.ring, (size_t)RR * HD * 2, cudaMemcpyDeviceToDevice));
                        CK(cudaMemcpy(C.comp, fin.comp, (size_t)CR * HD * 2, cudaMemcpyDeviceToDevice));
                        CK(cudaMemcpy(S.q, q_seq, (size_t)M * QW * 2, cudaMemcpyDeviceToDevice));
                        auto fn = [&] { attn_core(W, C, comp_src, S, M, 0, HEADS, K.ratio ? d_tkM : nullptr, TK, tk, st, d_pos); };
                        fn();
                        Verdict v;
                        v.add("o", cmp_bf16(S.o, o_seq, (long)M * QW));
                        const double us = tus(fn, reps);
                        char b[64];
                        snprintf(b, sizeof b, " fut_rows=%ld fut_max=%ld", fut_total, fut_max);
                        emit("core", M, s, us, v.cls == 0 ? "yes" : v.cls == 1 ? "class" : "no", K, gv, v.cls == 0, sq_core, v.detail + b);
                        if (cpa)
                        {   // k_attn_cpa on the same inputs: the o of the single-row k_attn reference must come back bit for bit
                            g_attn_cpa = 1;
                            CK(cudaMemset(S.o, 0xFF, (size_t)M * QW * 2));
                            fn();
                            Verdict vc;
                            vc.add("o", cmp_bf16(S.o, o_seq, (long)M * QW));
                            const double usc = tus(fn, reps);
                            g_attn_cpa = 0;
                            emit("core_cpa", M, s, usc, vc.cls == 0 ? "yes" : vc.cls == 1 ? "class" : "no", K, gv, vc.cls == 0, sq_core, vc.detail + b);
                        }
                        if (K.ratio)
                        {   // control: a list that leaks the first future row must change the output
                            auto fl = [&] { attn_core(W, C, comp_src, S, M, 0, HEADS, d_tkL, TK, tk, st, d_pos); };
                            fl();
                            const long nd = rows_differ(S.o, o_seq, M, QW);
                            const double usl = tus(fl, reps);
                            char c[64];
                            snprintf(c, sizeof c, " rows_differ=%ld/%d", nd, M);
                            emit("canary-leak", M, s, usl, nd ? "no" : "yes", K, gv, nd > 0, sq_core, c);
                        }
                    }
                    // ---- out, teacher-forced on the reference's S.o
                    if (want("out"))
                    {
                        CK(cudaMemcpy(S.o, o_seq, (size_t)M * QW * 2, cudaMemcpyDeviceToDevice));
                        auto fn = [&] { attn_out(W, S, M, 0, HEADS, d_out, st, d_pos); };
                        fn();
                        Verdict v;
                        v.add("out", cmp_bf16(d_out, out_seq, (long)M * HID));
                        const double us = tus(fn, reps);
                        emit("out", M, s, us, v.cls == 0 ? "yes" : v.cls == 1 ? "class" : "no", K, gv, v.cls == 0 || (v.cls == 1 && lenient), sq_out, v.detail);
                    }
                    // ---- layer: everything chained from y
                    if (want("layer"))
                    {
                        load(snaps[si]);
                        if (K.ratio == 1) CK(cudaMemcpy(C.comp + (size_t)s * HD, d_master + (size_t)s * HD, (size_t)M * HD * 2, cudaMemcpyDeviceToDevice));   // the indexer writes the chunk's rows first
                        auto fn = [&] { layer_forward(W, C, comp_src, S, d_y, M, 0, HEADS, K.ratio ? d_tkM : nullptr, TK, tk, d_out, st, d_pos); };
                        fn();
                        Verdict v;
                        v.add("out", cmp_bf16(d_out, out_seq, (long)M * HID));
                        if (K.ratio == 2) v.add("comp", cmp_bf16(C.comp, fin.comp, (long)CR * HD));
                        const double us = tus(fn, reps);
                        emit("layer", M, s, us, v.cls == 0 ? "yes" : v.cls == 1 ? "class" : "no", K, gv, v.cls == 0 || (v.cls == 1 && lenient), sq_layer, v.detail);
                    }
                }
    }
    g_use_gv = false;
    printf("# mgate: %d lines, %d not ok -> %s\n", lines, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
