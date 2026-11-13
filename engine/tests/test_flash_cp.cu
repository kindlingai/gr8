#include "../dispram.h"
// Flash context-parallel decode (attn/flash_attn.cu, G6_CP) against the replicated path, on random data. One GPU, every
// rank of TP = G6_NR on it, each with a replicated binding and a CP binding (R below), the exchange in device buffers.
// Steps at contexts below R, at R and past it (all pools visible to a row at most 512, one key tile, several), 1 to 16
// rows, in the plain form and the graph form (a step cell):
//   cache   every page a CP binding holds (the prefix, its own pages past R) equals the replicated binding's page,
//           latent and pooled keys, byte for byte
//   index   the merged lists equal the replicated lists: the same pools in the same order, the same tail
//   output  every rank's head group within the LSE merge's rounding of the replicated output (the partials are summed
//           in another order, so not the same bits), and the same bits in the plain and the graph form
//   below R a CP binding takes the replicated path: the same output bits as the replicated binding
//   groups  g6a_cp_groups (every head group in one call, equal groups) gives each group's g6a_cp_group partial, bit for bit
// Build & run on a Spark (sm_121a), in the toolchain container (KP_TILE_POOLS=1024 makes 4096-token key tiles, so the
// tiles run at small contexts; drop it for the shipped tile):
//   nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG6_KV4=1 -DG6_CP=1 -DDEC_S2 \
//     -DG6_NR=2 -DKP_TILE_POOLS=1024 -I engine engine/tests/test_flash_cp.cu engine/knobs-rs/target/release/libgr8knobs.a \
//     -lpthread -ldl -o /tmp/test_flash_cp && /tmp/test_flash_cp
// Exit code: 0 PASS, 1 FAIL, 2 CUDA error.
#include "../attn/flash_attn.cu"
#include "../attn/flash_bind.cu"   // flash_attn.cu calls the binder (g6a_bind_run, g6a_bind_prepare)
#include "../attn/kvchunks.cu"    // and the binder the chunked regions
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if !defined(G6_CP) || !G6_CP
#error "build with -DG6_CP=1"
#endif

#define CK(x)                                                                                                  \
    do {                                                                                                       \
        cudaError_t e_ = (x);                                                                                  \
        if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } \
    } while (0)
#define CKI(x)                                                                                                 \
    do {                                                                                                       \
        const int e_ = (x);                                                                                    \
        if (e_ != 0) { fprintf(stderr, "error %d from %s at %s:%d\n", e_, #x, __FILE__, __LINE__); exit(2); }   \
    } while (0)

static int g_fail = 0;
static void report(bool ok, const std::string& what) { printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str()); fflush(stdout); if (!ok) ++g_fail; }
template <class T> static T* dalloc(size_t n) { T* p = nullptr; CK(cudaMalloc(&p, n * sizeof(T) + 256)); CK(cudaMemset(p, 0, n * sizeof(T) + 256)); return p; }
template <class T> static void h2d(T* d, const std::vector<T>& h) { CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
template <class T> static std::vector<T> d2h(const T* d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 777) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uni() { return (float)((next() >> 40) * (1.0 / (1ull << 24))) * 2.f - 1.f; }
};
static std::vector<__nv_bfloat16> bf(size_t n, Rng& r, float scale) {
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) x = __float2bfloat16_rn(r.uni() * scale);
    return v;
}
static std::vector<__half> hf(size_t n, Rng& r, float scale) {
    std::vector<__half> v(n);
    for (auto& x : v) x = __float2half_rn(r.uni() * scale);
    return v;
}
template <class T> static T* upload(const std::vector<T>& v) { T* p = dalloc<T>(v.size()); h2d(p, v); return p; }
static G6AttnTensor tensor(const void* p, std::initializer_list<int64_t> dims) {
    G6AttnTensor t{};
    t.data = p; t.ndim = (int)dims.size();
    int j = 0;
    for (auto d : dims) t.shape[j++] = d;
    return t;
}

constexpr int NR = g6nr::NR, LAYER = 3, CAP = 24576, R = 1024, MR = 64, QT = 32, TPAGE = g6kv::PAGE;
constexpr int IH = g6kv::IDX_HEADS, ID = g6kv::IDX_DIM;
static int heads_of(int rank) { return g6nr::width(g6nr::HEADS, (rank - LAYER % NR + NR) % NR); }
static size_t blk(int M, int H) { return ((size_t)M * H * (fa::LAT + 2) + 3) & ~(size_t)3; }
// the CP binding's physical page of logical page lp on `rank` (-1: another rank's), flash_bind.cu's rule
static int phys_of(int lp, int rank) {
    if (lp < R / TPAGE) return lp;
    if (lp % NR != rank) return -1;
    int p = R / TPAGE;
    for (int q = R / TPAGE; q < lp; ++q) p += q % NR == rank;
    return p;
}
static float h2f(__half h) { return __half2float(h); }

struct Rank {
    G6AttnPack pack{};
    G6AttnBinding rep{}, cp{};
    int heads = 0;
    __half* qb = nullptr;     // [MR][heads][256]
    float* q = nullptr;       // the absorbed query, gathered: [MR][heads][512]
    float* part = nullptr;    // the partials of this rank's heads, [NR] blocks
    G6AttnStepCell* cell = nullptr;
};

int main() {
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    Rng wr(11);
    // the layer's weights every rank shares, and each rank's head group's UK / UV
    const auto kv_norm = upload(bf(geom::KV_LORA, wr, 1.f)), inorm = upload(bf(ID, wr, 1.f));
    const auto ibias = upload(bf(ID, wr, 0.1f)), ape = upload(bf((size_t)g6kv::KPOOL * ID, wr, 0.5f));
    std::vector<Rank> rk(NR);
    for (int r = 0; r < NR; ++r) {
        Rank& x = rk[r];
        const int h = x.heads = heads_of(r);
        x.pack.uk = tensor(upload(bf((size_t)h * geom::QK_NOPE * geom::KV_LORA, wr, 0.05f)), {h, geom::QK_NOPE, geom::KV_LORA});
        x.pack.uv = tensor(upload(bf((size_t)h * geom::V_HEAD * geom::KV_LORA, wr, 0.05f)), {h, geom::V_HEAD, geom::KV_LORA});
        x.pack.kv_norm = tensor(kv_norm, {geom::KV_LORA});
        x.pack.index_norm = tensor(inorm, {ID});
        x.pack.index_bias = tensor(ibias, {ID});
        x.pack.index_ape = tensor(ape, {g6kv::KPOOL, ID});
        CKI(g6a_bind_create_ext(&x.rep, &x.pack, r, LAYER, CAP, MR, QT, st, nullptr, 0));
        CKI(g6a_bind_create_cp(&x.cp, &x.pack, r, LAYER, CAP, MR, QT, R, NR, st, nullptr, 0));
        x.qb = dalloc<__half>((size_t)MR * h * 256);
        x.q = dalloc<float>((size_t)MR * h * fa::LAT);
        x.part = dalloc<float>(NR * blk(MR, h));
        x.cell = dalloc<G6AttnStepCell>(1);
    }
    printf("TP=%d, R %d, capacity %d: CP binding %d pages of %d on rank 0 (%.1f MB vs %.1f MB of cache)\n", NR, R, CAP,
           rk[0].cp.args.page_count, rk[0].rep.args.page_count,
           rk[0].cp.args.page_count * (TPAGE * g6kv::ROW + g6kv::IDX_PAGE) / 1e6,
           rk[0].rep.args.page_count * (TPAGE * g6kv::ROW + g6kv::IDX_PAGE) / 1e6);
    // every position's appended inputs (kv_a, raw index key, pool gate), the same for every binding
    Rng ir(23);
    __half* kv_a = upload(hf((size_t)CAP * geom::KV_LORA, ir, 1.f));
    __half* ik = upload(hf((size_t)CAP * ID, ir, 1.f));
    __half* ig = upload(hf((size_t)CAP * ID, ir, 1.f));
    std::vector<int> pos_h(CAP);
    for (int i = 0; i < CAP; ++i) pos_h[i] = i;
    int* pos = upload(pos_h);
    __half* iq = dalloc<__half>((size_t)MR * IH * ID);
    __half* iw = dalloc<__half>((size_t)MR * IH);
    uint64_t* cand = dalloc<uint64_t>((size_t)NR * MR * g6kv::TOP_POOLS);

    auto fill = [&](G6AttnBinding& b, int from, int to) {
        for (int p = from; p < to; p += MR) {
            G6Attn a = b.args;
            a.rows = std::min(MR, to - p); a.context = p + a.rows;
            a.kv_a = kv_a + (size_t)p * geom::KV_LORA; a.index_k = ik + (size_t)p * ID; a.index_gate = ig + (size_t)p * ID;
            a.positions = pos + p; a.raw_sess = nullptr;
            CKI(g6a_append(&a, st));
        }
    };
    auto step_args = [&](G6AttnBinding& b, const Rank& x, int ctx, int M) {
        G6Attn a = b.args;
        a.rows = M; a.context = ctx;
        a.q_b = x.qb; a.kv_a = kv_a + (size_t)(ctx - M) * geom::KV_LORA; a.positions = pos + ctx - M;
        a.index_q = iq; a.index_k = ik + (size_t)(ctx - M) * ID; a.index_weight = iw; a.index_gate = ig + (size_t)(ctx - M) * ID;
        a.raw_sess = nullptr; a.reuse_indices = nullptr; a.reuse_row_map = nullptr; a.reuse_rows = 0;
        return a;
    };
    auto err_of = [&](const G6AttnBinding& b) { return d2h(b.args.error, 1)[0]; };

    struct Case { int ctx, M; };
    const Case cases[] = {{900, 1}, {1024, 4}, {1600, 4}, {3001, 1}, {9000, 16}, {20003, 8}, {24576, 3}};
    int filled = 0;
    bool nonzero = false;
    for (const Case& c : cases) {
        const int ctx = c.ctx, M = c.M;
        const std::string tag = "ctx " + std::to_string(ctx) + " M " + std::to_string(M);
        for (Rank& x : rk) { fill(x.rep, filled, ctx - M); fill(x.cp, filled, ctx - M); }
        filled = ctx;
        // the step's queries
        Rng qr(1000003ull * ctx + M);
        h2d(iq, hf((size_t)M * IH * ID, qr, 1.f));
        h2d(iw, hf((size_t)M * IH, qr, 1.f));
        for (Rank& x : rk) h2d(x.qb, hf((size_t)M * x.heads * 256, qr, 1.f));
        // replicated reference
        std::vector<std::vector<__half>> ref(NR);
        std::vector<std::vector<int>> ref_ix(NR);
        for (int r = 0; r < NR; ++r) {
            Rank& x = rk[r];
            CK(cudaMemsetAsync(x.rep.args.error, 0, 4, st));
            G6Attn a = step_args(x.rep, x, ctx, M);
            CKI(g6a_run(&a, x.rep.scratch, x.rep.scratch_bytes, st));
            CK(cudaStreamSynchronize(st));
            ref[r] = d2h((const __half*)x.rep.args.output, (size_t)M * x.heads * 256);
            ref_ix[r] = d2h(x.rep.args.indices, (size_t)M * g6kv::SEL);
            for (__half v : ref[r]) nonzero |= h2f(v) != 0.f;
            if (err_of(x.rep)) report(false, tag + ": replicated error word " + std::to_string(err_of(x.rep)));
        }
        if (ctx <= R) {
            // below R: the CP binding runs the replicated path
            bool same = true;
            for (int r = 0; r < NR; ++r) {
                Rank& x = rk[r];
                CK(cudaMemsetAsync(x.cp.args.error, 0, 4, st));
                G6Attn a = step_args(x.cp, x, ctx, M);
                CKI(g6a_run(&a, x.cp.scratch, x.cp.scratch_bytes, st));
                CK(cudaStreamSynchronize(st));
                const auto o = d2h((const __half*)x.cp.args.output, (size_t)M * x.heads * 256);
                same &= !memcmp(o.data(), ref[r].data(), o.size() * 2) && err_of(x.cp) == 0;
            }
            report(same, tag + ": below R, the CP binding's replicated path gives the replicated bits");
            continue;
        }
        std::vector<std::vector<__half>> out_plain(NR);
        for (int form = 0; form < 2; ++form) {   // 0 plain, 1 graph (step cell, captured capacity CAP)
            std::vector<G6Attn> a(NR);
            for (int r = 0; r < NR; ++r) {
                Rank& x = rk[r];
                a[r] = step_args(x.cp, x, ctx, M);
                CK(cudaMemsetAsync(x.cp.args.error, 0, 4, st));
                if (form) {
                    a[r].context = CAP;
                    G6AttnStepCell cl{};
                    cl.context = ctx; cl.page_count = x.cp.args.page_count; cl.table_size = (ctx + TPAGE - 1) / TPAGE;
                    CK(cudaMemcpyAsync(x.cell, &cl, sizeof cl, cudaMemcpyHostToDevice, st));
                }
            }
            auto cell = [&](int r) { return form ? (const G6AttnStepCell*)rk[r].cell : nullptr; };
            CK(cudaMemsetAsync(cand, 0, (size_t)NR * MR * g6kv::TOP_POOLS * 8, st));
            for (int r = 0; r < NR; ++r) {
                Rank& x = rk[r];
                CKI(g6a_cp_front(&a[r], cell(r), x.cp.scratch, x.cp.scratch_bytes, st));
                const float *q = nullptr, *rope = nullptr;
                CKI(g6a_cp_qabs(&a[r], x.cp.scratch, &q, &rope));
                if (rope) report(false, "g6a_cp_qabs: Flash has no RoPE part");
                CK(cudaMemcpyAsync(x.q, q, (size_t)M * x.heads * fa::LAT * 4, cudaMemcpyDeviceToDevice, st));
            }
            for (int r = 0; r < NR; ++r) CKI(g6a_cp_index(&a[r], cell(r), rk[r].cp.scratch, rk[r].cp.scratch_bytes, cand, st));
            for (int r = 0; r < NR; ++r) CKI(g6a_cp_merge_index(&a[r], cell(r), cand, st));
            for (int s = 0; s < NR; ++s)        // the keys rank s scores, for every head group d
                for (int d = 0; d < NR; ++d) {
                    G6Attn ad = a[s];
                    ad.heads = rk[d].heads;
                    CKI(g6a_cp_group(&ad, cell(s), rk[s].cp.scratch, rk[s].cp.scratch_bytes, rk[d].q, nullptr,
                                     rk[d].part + s * blk(M, rk[d].heads), st));
                }
            for (int d = 0; d < NR; ++d) CKI(g6a_cp_finish(&a[d], cell(d), rk[d].cp.scratch, rk[d].cp.scratch_bytes, rk[d].part, st));
            CK(cudaStreamSynchronize(st));
            if (g6nr::HMIN == g6nr::HMAX && g6a_cp_groups_ok(rk[0].heads)) {
                // every group's partial over rank s's keys in one call: group d at heads d H..
                const int H = rk[0].heads;
                const size_t gb = blk(M, H), fb = (size_t)M * H * (fa::LAT + 2) * 4;
                float* qall = dalloc<float>((size_t)M * NR * H * fa::LAT);
                float* fused = dalloc<float>(NR * gb);
                const size_t bytes = g6a_cp_groups_bytes(NR * H, M);
                char* buf = dalloc<char>(bytes);
                bool same = true;
                for (int s = 0; s < NR; ++s) {
                    for (int d = 0; d < NR; ++d)
                        CK(cudaMemcpy2DAsync(qall + (size_t)d * H * fa::LAT, (size_t)NR * H * fa::LAT * 4, rk[d].q, (size_t)H * fa::LAT * 4,
                                             (size_t)H * fa::LAT * 4, M, cudaMemcpyDeviceToDevice, st));
                    G6Attn ad = a[s];
                    ad.heads = NR * H;
                    CKI(g6a_cp_groups(&ad, cell(s), buf, bytes, qall, nullptr, fused, H, (int64_t)gb, st));
                    CK(cudaStreamSynchronize(st));
                    for (int d = 0; d < NR; ++d)
                        same &= d2h((const uint8_t*)(fused + d * gb), fb) == d2h((const uint8_t*)(rk[d].part + s * blk(M, H)), fb);
                }
                report(same, tag + " " + (form ? "graph" : "plain") + ": g6a_cp_groups gives every group's g6a_cp_group partial");
                cudaFree(qall); cudaFree(fused); cudaFree(buf);
            }
            const char* fn = form ? "graph" : "plain";
            bool ix_ok = true, err_ok = true, out_ok = true, same = true;
            double maxd = 0, maxr = 0;
            size_t ndiff = 0;
            for (int r = 0; r < NR; ++r) {
                Rank& x = rk[r];
                ix_ok &= d2h(x.cp.args.indices, (size_t)M * g6kv::SEL) == ref_ix[r];
                const unsigned e = err_of(x.cp);
                if (e) { err_ok = false; printf("  rank %d error word %#x\n", r, e); }
                const auto o = d2h((const __half*)x.cp.args.output, (size_t)M * x.heads * 256);
                double rmax = 0;
                for (__half v : ref[r]) rmax = std::max(rmax, (double)fabsf(h2f(v)));
                for (size_t j = 0; j < o.size(); ++j) {
                    const double d = fabs((double)h2f(o[j]) - h2f(ref[r][j]));
                    maxd = std::max(maxd, d);
                    ndiff += d != 0;
                    out_ok &= d <= 1e-3 * rmax && std::isfinite(h2f(o[j]));
                }
                maxr = std::max(maxr, rmax);
                if (!form) out_plain[r] = o;
                else same &= !memcmp(o.data(), out_plain[r].data(), o.size() * 2);
            }
            report(err_ok, tag + " " + fn + ": error words 0");
            report(ix_ok, tag + " " + fn + ": merged index lists equal the replicated lists");
            char buf[160];
            snprintf(buf, sizeof buf, ": output within LSE rounding (max |diff| %.3g of max |ref| %.3g, %zu values differ)", maxd, maxr, ndiff);
            report(out_ok, tag + " " + fn + buf);
            if (form) report(same, tag + ": graph form gives the plain form's bits");
        }
    }
    report(nonzero, "the outputs compared are not all zero");
    // every page a CP binding holds equals the replicated page
    bool cache_ok = true;
    const int pages = (filled + TPAGE - 1) / TPAGE;
    for (int r = 0; r < NR; ++r) {
        const G6Attn& c = rk[r].cp.args;
        const G6Attn& p = rk[r].rep.args;
        int held = 0;
        for (int lp = 0; lp < pages; ++lp) {
            const int ph = phys_of(lp, r);
            if (ph < 0) continue;
            ++held;
            cache_ok &= d2h((const uint8_t*)c.mla_cache + (size_t)ph * TPAGE * g6kv::ROW, TPAGE * g6kv::ROW) ==
                        d2h((const uint8_t*)p.mla_cache + (size_t)lp * TPAGE * g6kv::ROW, TPAGE * g6kv::ROW);
            cache_ok &= d2h((const uint8_t*)c.index_cache + (size_t)ph * g6kv::IDX_PAGE, g6kv::IDX_PAGE) ==
                        d2h((const uint8_t*)p.index_cache + (size_t)lp * g6kv::IDX_PAGE, g6kv::IDX_PAGE);
        }
        cache_ok &= held <= c.page_count;
        cache_ok &= d2h((const uint8_t*)c.index_raw, g6kv::RAW_BYTES) == d2h((const uint8_t*)p.index_raw, g6kv::RAW_BYTES);
    }
    report(cache_ok, "every page a CP binding holds, and its raw ring, equal the replicated binding's");
    if (g6dispram::enabled()) {
        g6dispram::Client probe;
        g6dispram::Info dinfo{};
        const bool has_daemon = probe.open_conn(g6dispram::socket_path()) && probe.query_info(&dinfo);
        probe.close_conn();
        if (has_daemon || knob::dispram_min_bytes() > 0)
            report(g6dispram::claimed_bytes() > 0, "non-chunked Flash bindings claimed live dispramd carveout bytes");
    } else {
        report(g6dispram::claimed_bytes() == 0, "non-chunked Flash bindings stay off dispramd by default");
    }
    for (Rank& x : rk) { g6a_bind_release(&x.rep); g6a_bind_release(&x.cp); }
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
