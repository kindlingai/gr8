// The Inkling 2 Small speed paths on the device, each against the stock path it replaces.
//
//   1. The head's candidates packed on the device (kernels/ink2_head.cuh) against ink2rt::head_pack on the host:
//      rows of logits with -inf cells, runs of equal logits, both zeros, fewer finite logits than candidates and
//      none at all; 16, 256 and 512 candidates. The maximum, the count and every candidate: exact. The sum: to 1e-12.
//   2. The tensor-core attention of a prefill chunk (attn/ink2_attn_tc.cuh) against k_ink2_attn_core at the full
//      model's head geometry (head_dim 128, rel_dim 16): a local layer whose keys come from the ring and from the
//      chunk, a rank whose first K / V head has part of its query heads, and a global layer on pages. 40 rows (a
//      part group of 16). A cell may differ by max(|x|, 1) / 32 (the sums are in another order and BF16 rounds the
//      output); the largest difference is printed.
//   3. The engine on a miniature checkpoint (ink2_mini_ckpt.h), 1 and 4 simulated ranks, a 70-token prefill in
//      chunks of 48 and 22 with every row sampled, then 8 single rows: the device head pack on and off give the
//      same tokens and candidates, exact. With the cuBLAS projections on as well, a row's token is the stock
//      row's, or a stock candidate within max(|top|, 1) / 32 of its top (a near tie, counted and printed).
//      (The miniature's head_dim is 16, so the tensor-core kernel is not on this path; check 2 covers it.)
//   4. The device all-reduce (kernels/ink2_fabric.cuh) as rank 1 of a 4-rank fabric in pinned host memory, the test
//      playing the peers (before the launches, or after the doorbell while the device spins): 1, 8, 64 and 512 rows
//      of 2304 and 4096 cells; the sum's bits are the stock sum's. Also the slot offsets and the credits of the pooled
//      96-collective layout. The proxy and the wire of a real multi-rank run are not on this path.
//
// --bench: also times the head pack (8 rows of 50304 logits, 256 candidates) and the two attention kernels
// (512 rows, a local window of 1024) and prints milliseconds.
//
// Build (from engine/runtime, after scripts/build.sh has built the Rust libraries and bin/obj/g6pack.o):
//   nvcc -O3 -std=c++17 -Xcompiler -ffp-contract=off -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_INK2=1 -DDEC_S2 \
//     -DG6_RS_LOADER -I . -I ../kernels -I ../attn -I ../ref/ink2 -o ../../bin/ink2_speed_test ../tests/ink2_speed_test.cu \
//     ../../bin/obj/g6pack.o ../loader-rs/target/release/libg6load.a ../knobs-rs/target/release/libgr8knobs.a \
//     ../target/release/libg6host.a -lcublas -lpthread -ldl -lm
// Run: bin/ink2_speed_test [--bench]   (one GPU; prints PASS and returns 0)
#include "../comms/fabric.h"
#include "../kernels/ink2_fabric.cuh"
#include "../runtime/ink2_step.cuh"
#include "ink2_mini_ckpt.h"
#include <chrono>
#include <filesystem>
#include <memory>

using namespace ink2mini;
using ink2rt::Engine;
using ink2rt::Options;
using ink2rt::RowOut;
using ink2rt::RowSample;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) { fprintf(stderr, "FATAL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } \
    } while (0)

static int fails = 0;
static void check(bool ok, const std::string& what)
{
    if (ok) return;
    ++fails;
    if (fails <= 40) fprintf(stderr, "FAIL %s\n", what.c_str());
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint16_t f2bf(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
static float bf2f(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float unit(Rng& r) { return static_cast<float>(r.next() & 0xffffff) / 8388608.0f - 1.0f; }   // [-1, 1)
template <class T> static T* to_dev(const std::vector<T>& v)
{
    T* d = nullptr;
    CK(cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)));
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static std::vector<uint16_t> rand_bf(Rng& r, size_t n, float scale)
{
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = f2bf(unit(r) * scale);
    return v;
}

// ---------------------------------------------------------------------------------------------- 1. the head pack
static void check_head_pack(int vr, int NC, int rows, bool bench)
{
    Rng r{0x68656164u + static_cast<uint64_t>(vr) * 31 + static_cast<uint64_t>(NC)};
    std::vector<float> lg(static_cast<size_t>(rows) * vr);
    for (int m = 0; m < rows; ++m)
        for (int j = 0; j < vr; ++j)
        {
            float v = unit(r) * 12.0f;
            const uint32_t k = r.next();
            if (m % 4 == 1) v = static_cast<float>(static_cast<int>(v));           // many equal logits
            if (m % 4 == 2 && k % 3) v = -INFINITY;                                // a third finite
            if (m % 4 == 3 && j >= 5) v = -INFINITY;                               // fewer finite than candidates
            if (m == rows - 1) v = -INFINITY;                                      // none
            if (m % 4 == 1 && k % 97 == 0) v = k & 256 ? 0.0f : -0.0f;             // both zeros
            if (m == 0 && j % 1000 == 7) v = 11.5f;                                // a run of equal maxima across the threads' chunks
            lg[static_cast<size_t>(m) * vr + j] = v;
        }
    const size_t rb = ink2rt::head_row_bytes(NC);
    const int v0 = 3 * vr;
    std::vector<uint8_t> ref(static_cast<size_t>(rows) * rb), got(ref.size());
    double t0 = now_ms();
    for (int m = 0; m < rows; ++m) ink2rt::head_pack(&lg[static_cast<size_t>(m) * vr], vr, v0, NC, &ref[static_cast<size_t>(m) * rb]);
    const double t_cpu = now_ms() - t0;
    float* d_lg = to_dev(lg);
    uint8_t* d_out = to_dev(got);
    ink2::head_pack_dev(d_lg, rows, vr, v0, NC, d_out, rb, 0);
    CK(cudaDeviceSynchronize());
    t0 = now_ms();
    ink2::head_pack_dev(d_lg, rows, vr, v0, NC, d_out, rb, 0);
    CK(cudaMemcpy(got.data(), d_out, got.size(), cudaMemcpyDeviceToHost));
    const double t_gpu = now_ms() - t0;
    for (int m = 0; m < rows; ++m)
    {
        const uint8_t *a = &ref[static_cast<size_t>(m) * rb], *b = &got[static_cast<size_t>(m) * rb];
        const std::string w = "head pack vr " + std::to_string(vr) + " nc " + std::to_string(NC) + " row " + std::to_string(m);
        double sa, sb;
        memcpy(&sa, a + 8, 8);
        memcpy(&sb, b + 8, 8);
        check(memcmp(a, b, 8) == 0, w + ": maximum or count");
        check(std::fabs(sa - sb) <= 1e-12 * std::max(1.0, std::fabs(sa)), w + ": sum " + std::to_string(sa) + " / " + std::to_string(sb));
        check(memcmp(a + 16, b + 16, rb - 16) == 0, w + ": candidates");
    }
    if (bench) printf("   bench head pack: %d rows of %d logits, %d candidates: host %.3f ms (without its copy of the logits), device %.3f ms\n", rows, vr, NC, t_cpu, t_gpu);
    CK(cudaFree(d_lg));
    CK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------------------------- 2. the attention
struct AttnCase
{
    const char* name;
    int M, p0, nh_q, q0, kv0, nh_kv, window, global;
};
static void check_attn_tc(const AttnCase& tc, bool bench)
{
    const int hd = ink2::TC_HD, rd = ink2::TC_RD, gqa = 6, rel_win = 1024, base = 6, M = tc.M, nk = tc.nh_kv * hd;
    Rng r{0x7463u + static_cast<uint64_t>(tc.p0) * 131 + static_cast<uint64_t>(tc.nh_q)};
    const int pmax = tc.p0 + M - 1, max_pps = pmax / ink2::KV_PAGE_TOKENS + 1;
    std::vector<int> slot(static_cast<size_t>(M), 0), pos(static_cast<size_t>(M)), run0(static_cast<size_t>(M), 0), pt(static_cast<size_t>(max_pps));
    for (int m = 0; m < M; ++m) pos[m] = tc.p0 + m;
    for (int i = 0; i < max_pps; ++i) pt[i] = max_pps - 1 - i;   // the pages are not in position order
    const std::vector<uint16_t> q_raw = rand_bf(r, static_cast<size_t>(M) * tc.nh_q * hd, 1.5f), r_raw = rand_bf(r, static_cast<size_t>(M) * tc.nh_q * rd, 1.5f),
                                gate = rand_bf(r, static_cast<size_t>(M) * tc.nh_q * hd, 2.0f), sink = rand_bf(r, static_cast<size_t>(M) * tc.nh_q, 2.0f),
                                q_clip = rand_bf(r, static_cast<size_t>(tc.nh_q) * hd, 1.2f), rel_clip = rand_bf(r, static_cast<size_t>(tc.nh_q) * rd, 1.2f),
                                rel_proj = rand_bf(r, static_cast<size_t>(rd) * rel_win, 0.6f), kv_new = rand_bf(r, static_cast<size_t>(M) * 2 * nk, 0.5f),
                                ring = rand_bf(r, static_cast<size_t>(2) * tc.window * nk, 0.5f),
                                pages = rand_bf(r, static_cast<size_t>(max_pps) * 2 * ink2::KV_PAGE_TOKENS * nk, 0.5f);
    const int smax = ink2::attn_splits(tc.global != 0, tc.window, pmax);
    const size_t ncell = static_cast<size_t>(M) * tc.nh_q * hd;
    uint16_t *d_q = to_dev(q_raw), *d_r = to_dev(r_raw), *d_gate = to_dev(gate), *d_sink = to_dev(sink), *d_qc = to_dev(q_clip), *d_rc = to_dev(rel_clip), *d_rp = to_dev(rel_proj),
             *d_kv = to_dev(kv_new), *d_ring = to_dev(ring), *d_pages = to_dev(pages);
    int *d_slot = to_dev(slot), *d_pos = to_dev(pos), *d_run0 = to_dev(run0), *d_pt = to_dev(pt);
    uint16_t* d_og = to_dev(std::vector<uint16_t>(ncell, 0));
    float* d_part = to_dev(std::vector<float>(static_cast<size_t>(M) * tc.nh_q * smax * (hd + 2), 0.0f));
    ink2::AttnCoreArgs a{d_q, d_r, d_gate, d_sink, d_qc, d_rc, d_rp, d_kv, d_ring, d_pages, d_pt, ink2::Rows{d_slot, d_pos, d_run0}, d_og, d_part,
                         0, M, tc.nh_q, tc.q0, gqa, tc.kv0, nk, hd, rd, rel_win, tc.window, base, tc.global, max_pps, smax};
    std::vector<uint16_t> ref(ncell), got(ncell);
    CK(cudaDeviceSynchronize());
    double t0 = now_ms();
    ink2::k_ink2_attn_core<<<dim3(M * tc.nh_q, smax), ink2::ACT>>>(a);
    if (smax > 1) ink2::k_ink2_attn_merge<<<M * tc.nh_q, ink2::ACT>>>(a);
    CK(cudaDeviceSynchronize());
    const double t_ref = now_ms() - t0;
    CK(cudaMemcpy(ref.data(), d_og, ncell * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemset(d_og, 0, ncell * 2));
    CK(cudaFuncSetAttribute(ink2::k_ink2_attn_prefill_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(ink2::attn_tc_smem(ink2::TC_MAX_GQ))));
    t0 = now_ms();
    ink2::k_ink2_attn_qr_prep<<<M * tc.nh_q, ink2::ACT>>>(d_q, d_r, d_qc, d_rc, a.rw, tc.nh_q, hd, rd, base, tc.global);
    ink2::k_ink2_attn_prefill_tc<<<dim3((M + 15) / 16, tc.nh_kv), gqa * 32, ink2::attn_tc_smem(gqa)>>>(a, gqa);
    CK(cudaDeviceSynchronize());
    const double t_tc = now_ms() - t0;
    CK(cudaMemcpy(got.data(), d_og, ncell * 2, cudaMemcpyDeviceToHost));
    double worst = 0;
    size_t bad = 0, differ = 0;
    for (size_t i = 0; i < ncell; ++i)
    {
        const float x = bf2f(ref[i]), y = bf2f(got[i]);
        const double d = std::fabs(static_cast<double>(x) - y), tol = std::max(1.0, static_cast<double>(std::fabs(x))) / 32;
        if (!(d <= tol)) ++bad;   // a NaN is a failure
        if (ref[i] != got[i]) ++differ;
        if (d > worst) worst = d;
    }
    printf("   %s: %zu cells, %zu differ, largest difference %.5f%s\n", tc.name, ncell, differ, worst, bad ? "  <-- FAIL" : "");
    check(bad == 0, std::string("tensor-core attention, ") + tc.name + ": " + std::to_string(bad) + " cells past the tolerance");
    if (bench) printf("   bench attention %s: %d rows, stock %.3f ms, tensor cores %.3f ms\n", tc.name, M, t_ref, t_tc);
    for (void* p : {static_cast<void*>(d_q), static_cast<void*>(d_r), static_cast<void*>(d_gate), static_cast<void*>(d_sink), static_cast<void*>(d_qc), static_cast<void*>(d_rc),
                    static_cast<void*>(d_rp), static_cast<void*>(d_kv), static_cast<void*>(d_ring), static_cast<void*>(d_pages), static_cast<void*>(d_slot), static_cast<void*>(d_pos),
                    static_cast<void*>(d_run0), static_cast<void*>(d_pt), static_cast<void*>(d_og), static_cast<void*>(d_part)})
        CK(cudaFree(p));
}

// ---------------------------------------------------------------------------------------------- 3. the engine
constexpr int CAP = 192, ROWS = 48, NCAND = 64, PROMPT = 70, SINGLES = 8;

static std::unique_ptr<Engine> engine(const std::string& dir, int T, int ghp, int blas)
{
    Options o;
    o.nr = T;
    o.rank0 = 0;
    o.nloc = T;
    o.slots = 1;
    o.max_rows = ROWS;
    o.cap = CAP;
    o.nc = NCAND;
    o.fast_allreduce = 0;
    o.gpu_head_pack = ghp;
    o.cublas_prefill = blas;
    o.tc_prefill_attn = blas;
    std::unique_ptr<Engine> e(new Engine);
    e->init(dir, o);
    return e;
}
// every row's output of the teacher-forced run: the prompt in chunks of ROWS, then single rows
static std::vector<RowOut> run(Engine& e, const std::vector<int>& toks)
{
    std::vector<RowOut> all, out;
    e.reset_slot(0);
    int p = 0;
    const int P = static_cast<int>(toks.size());
    while (p < P)
    {
        const int n = p < PROMPT ? std::min(ROWS, PROMPT - p) : 1;
        ink2::RowsHost rh;
        rh.add(0, p, n);
        std::vector<RowSample> sm;
        for (int i = 0; i < n; ++i) sm.push_back(RowSample{0, 0, 0, static_cast<uint32_t>(p + i + 1)});
        e.forward(rh, &toks[static_cast<size_t>(p)], sm.data(), out);
        e.commit();
        all.insert(all.end(), out.begin(), out.end());
        p += n;
    }
    return all;
}
static void check_engine(const std::string& dir, int T, const std::vector<int>& toks)
{
    const std::vector<RowOut> off = run(*engine(dir, T, 0, 0), toks), ghp = run(*engine(dir, T, 1, 0), toks), all = run(*engine(dir, T, 1, 1), toks);
    const std::string w = "engine of " + std::to_string(T) + " rank(s), row ";
    check(off.size() == toks.size() && ghp.size() == off.size() && all.size() == off.size(), w + "count");
    if (fails) return;
    int near = 0;
    for (size_t i = 0; i < off.size(); ++i)
    {
        const RowOut &a = off[i], &b = ghp[i], &c = all[i];
        const std::string wi = w + std::to_string(i);
        check(a.tok == b.tok, wi + ": the device head pack changed the token");
        check(std::fabs(a.lse - b.lse) <= 1e-5f * std::max(1.0f, std::fabs(a.lse)), wi + ": the device head pack changed the log-sum-exp");
        check(a.top.size() == b.top.size() && (a.top.empty() || memcmp(a.top.data(), b.top.data(), a.top.size() * sizeof(ink2rt::Cand)) == 0),
              wi + ": the device head pack changed the candidates");
        if (a.top.empty() || c.top.empty()) { check(a.top.empty() == c.top.empty(), wi + ": candidates with cuBLAS"); continue; }
        const float tol = std::max(1.0f, std::fabs(a.top[0].logit)) / 32;
        check(std::fabs(a.top[0].logit - c.top[0].logit) <= tol, wi + ": top logit with cuBLAS " + std::to_string(c.top[0].logit) + ", stock " + std::to_string(a.top[0].logit));
        if (c.tok == a.tok) continue;
        bool tie = false;
        for (const ink2rt::Cand& x : a.top)
            if (x.id == c.tok) tie = a.top[0].logit - x.logit <= tol;
        check(tie, wi + ": token with cuBLAS " + std::to_string(c.tok) + ", stock " + std::to_string(a.tok) + ", not a near tie");
        near += tie;
    }
    printf("   %d rank(s): %zu rows; device head pack exact; with cuBLAS %d near tie(s)\n", T, off.size(), near);
}

// ------------------------------------------------------------------------------------- 4. the device all-reduce
// A fabric of 4 ranks in pinned host memory with no proxy: this test is rank 1 and plays the three peers. Even cases
// have the peers' shares and tail words in the receive slots before the launches; in odd cases the host waits for the
// doorbell, then writes the shares and, last, the tail words, while the device spins.
template <class T> static T* pinned(size_t n)
{
    void* p = nullptr;
    CK(cudaHostAlloc(&p, std::max<size_t>(1, n) * sizeof(T), cudaHostAllocMapped));
    memset(p, 0, std::max<size_t>(1, n) * sizeof(T));
    return static_cast<T*>(p);
}
static void check_fabric()
{
    const int NR = 4, ME = 1, NCOLL = 96;
    const size_t max_bytes = static_cast<size_t>(512) * 4096 * 4;
    g6::FabricConfig cfg;
    cfg.slot_bytes = max_bytes;
    cfg.pool = 2;
    cfg.run1 = cfg.run2 = NCOLL;
    cfg.slot_bytes1 = 8;
    g6::FabricDev f{};
    f.rank = ME;
    f.nr = NR;
    if (!g6::fab_slot_layout(cfg, NCOLL, &f)) { check(false, "fabric: the pooled layout is refused"); return; }
    const size_t area = g6::fab_slot_base(f, NCOLL);
    check(f.pool == 2 && area == 2 * f.stride && g6::fab_recv_bytes(f) == 2 * area * NR, "fabric: the pooled areas");
    bool lay = true;
    for (int par = 0; par < 2; ++par)
        for (int c = 0; c < NCOLL; ++c)
        {
            lay = lay && g6::fab_pool_slot(f, c) == c % 2 && g6::fab_send_off(f, par, c) == par * area + (c % 2) * f.stride;
            for (int p = 0; p < NR; ++p) lay = lay && g6::fab_recv_off(f, par, c, p) == (par * area + (c % 2) * f.stride) * NR + p * f.stride;
        }
    check(lay, "fabric: the slot offsets of the pooled layout");
    std::vector<int> walk(NCOLL), credit;
    for (int c = 0; c < NCOLL; ++c) walk[c] = c;
    bool cr = g6::fab_pool_credits(f, walk, &credit) && credit[0] == -1 && credit[1] == -1;
    for (int c = 2; cr && c < NCOLL; ++c) cr = credit[c] == c - 1;
    check(cr, "fabric: the pool credits of a step");

    f.send = pinned<char>(2 * area);
    f.recv = pinned<char>(g6::fab_recv_bytes(f));
    f.doorbell = pinned<unsigned long long>(NCOLL);
    f.error = pinned<unsigned>(1);
    Rng r{0x66616272u};
    int k = 0;
    for (int M : {1, 8, 64, 512})
        for (int cols : {2304, 4096})
        {
            const size_t n = static_cast<size_t>(M) * cols, bytes = n * 4;
            const unsigned long long step = 3 + k;
            const int c = (k * 13 + 5) % NCOLL, par = static_cast<int>(step & 1);
            const unsigned long long seq = g6::coll_seq(step, c, NCOLL);
            const std::string tag = "fabric: M " + std::to_string(M) + " cols " + std::to_string(cols) + ": ";
            std::vector<float> parts(NR * n), mine(NR * n);
            for (float& x : parts) x = unit(r) * 3.0f;
            memset(mine.data(), 0xff, mine.size() * 4);   // the peers' places hold NaN until the exchange fills them
            memcpy(mine.data() + ME * n, parts.data() + ME * n, bytes);
            float* d_ref = to_dev(parts);
            float* d_mine = to_dev(mine);
            uint16_t *o_ref = nullptr, *o_mine = nullptr;
            CK(cudaMalloc(&o_ref, n * 2));
            CK(cudaMalloc(&o_mine, n * 2));
            ink2::sum_ranks(d_ref, NR, n, o_ref, 0);
            CK(cudaStreamSynchronize(0));
            auto peers = [&](bool tails) {
                for (int p = 0; p < NR; ++p)
                {
                    if (p == ME) continue;
                    char* slot = f.recv + g6::fab_recv_off(f, par, c, p);
                    if (!tails) memcpy(slot, parts.data() + p * n, bytes);
                    else __atomic_store_n(reinterpret_cast<unsigned long long*>(slot + bytes), seq, __ATOMIC_SEQ_CST);
                }
            };
            const bool late = k & 1;
            if (!late) { peers(false); peers(true); }
            ink2::ar_exchange(f, step, c, reinterpret_cast<char*>(d_mine), bytes, 0);
            ink2::sum_ranks(d_mine, NR, n, o_mine, 0);
            if (late)
            {
                const double t0 = now_ms();
                while (__atomic_load_n(f.doorbell + c, __ATOMIC_SEQ_CST) != seq && now_ms() - t0 < 20000.0) {}
                check(__atomic_load_n(f.doorbell + c, __ATOMIC_SEQ_CST) == seq, tag + "no doorbell in 20 s");
                peers(false);
                peers(true);
            }
            CK(cudaStreamSynchronize(0));
            std::vector<uint16_t> a(n), b(n);
            CK(cudaMemcpy(a.data(), o_ref, n * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), o_mine, n * 2, cudaMemcpyDeviceToHost));
            check(memcmp(a.data(), b.data(), n * 2) == 0, tag + "the sum is not the stock sum's bits");
            check(f.doorbell[c] == seq, tag + "the doorbell word");
            check(memcmp(f.send + g6::fab_send_off(f, par, c), parts.data() + ME * n, bytes) == 0, tag + "the send slot");
            check(*f.error == 0, tag + "the error word is set");
            CK(cudaFree(d_ref));
            CK(cudaFree(d_mine));
            CK(cudaFree(o_ref));
            CK(cudaFree(o_mine));
            ++k;
        }
    printf("   %d collectives of 4 ranks: the sum's bits are the stock sum's\n", k);
    CK(cudaFreeHost(f.send));
    CK(cudaFreeHost(f.recv));
    CK(cudaFreeHost(f.doorbell));
    CK(cudaFreeHost(f.error));
}

int main(int argc, char** argv)
{
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    knob::init();
    printf("1. the head's candidates packed on the device\n");
    for (int nc : {16, 256, 512})
    {
        check_head_pack(1000, nc, 9, false);
        check_head_pack(12576, nc, 9, false);
    }
    check_head_pack(255, 256, 5, false);
    if (bench) check_head_pack(50304, 256, 8, true);

    printf("2. the tensor-core attention of a prefill chunk\n");
    check_attn_tc({"local, ring and chunk", 40, 200, 12, 0, 0, 2, 96, 0}, false);
    check_attn_tc({"local, split K / V head", 40, 37, 9, 3, 0, 2, 96, 0}, false);
    check_attn_tc({"local, first positions", 40, 0, 12, 0, 0, 2, 96, 0}, false);
    check_attn_tc({"global, pages", 40, 100, 12, 6, 1, 2, 96, 1}, false);
    check_attn_tc({"global, past the rel window", 24, 1100, 6, 0, 0, 1, 96, 1}, false);
    if (bench) check_attn_tc({"local, window 1024", 512, 2048, 12, 0, 0, 2, 1024, 0}, true);

    printf("3. the engine with the speed paths on and off\n");
    Mini mm;
    build_mini(mm);
    char tmpl[] = "/tmp/ink2-speed-test.XXXXXX";
    if (!mkdtemp(tmpl)) { perror("mkdtemp"); return 2; }
    const std::string tmp = tmpl, dir = tmp + "/ckpt";
    write_ckpt(mm, dir);
    Rng r{0x73706565u};
    std::vector<int> toks;
    for (int i = 0; i < PROMPT + SINGLES; ++i) toks.push_back(static_cast<int>(r.next() % VALID));
    for (int T : {1, 4}) check_engine(dir, T, toks);
    std::filesystem::remove_all(tmp);

    printf("4. the device all-reduce on a fabric in memory\n");
    check_fabric();

    if (fails)
    {
        printf("FAIL (%d)\n", fails);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
