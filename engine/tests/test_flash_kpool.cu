// The Flash kpool indexer (attn/kpool.cuh), two exactness checks on random data:
//   pools   the pooled keys of calls of any size (1 row up to a 4096-row prefill quantum), at any position, with
//           speculation rollbacks (rows written again with new values) and two sessions interleaved, each reading the
//           raw rows before its call from its raw ring, equal byte for byte the keys of one call over the final rows
//   tiles   kp_select_tile over key tiles (KP_TILE pools, built small here) gives kp_select's lists, ties included,
//           after the decode scorer (kp_scores) and the prefill tensor-core scorer (kp_scores_tc), whose per-pool
//           scores with a pool offset equal the one-pass scores
// Build & run on a Spark (sm_121a), in the toolchain container:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG6_KV4=1 -DDEC_S2 -DG6_NR=1 \
//     -DKP_TILE_POOLS=1024 -I engine engine/tests/test_flash_kpool.cu engine/knobs-rs/target/release/libgr8knobs.a \
//     -lpthread -ldl -o /tmp/test_flash_kpool && /tmp/test_flash_kpool
#include "../attn/flash_attn.cu"
#include "../attn/flash_bind.cu"   // flash_attn.cu calls the binder (g6a_bind_run, g6a_bind_prepare)
#include "../attn/kvchunks.cu"    // and the binder the chunked regions
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                                  \
    do {                                                                                                       \
        cudaError_t e_ = (x);                                                                                  \
        if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } \
    } while (0)

static int g_fail = 0;
static void report(bool ok, const char* what) { printf("%s %s\n", ok ? "PASS" : "FAIL", what); if (!ok) ++g_fail; }
template <class T> static T* dalloc(size_t n) { T* p = nullptr; CK(cudaMalloc(&p, n * sizeof(T) + 256)); CK(cudaMemset(p, 0, n * sizeof(T) + 256)); return p; }
template <class T> static void h2d(T* d, const std::vector<T>& h) { CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
template <class T> static std::vector<T> d2h(const T* d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
static uint16_t f2h(float f) { __half h = __float2half_rn(f); uint16_t u; memcpy(&u, &h, 2); return u; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16); }

// one session's view: its own pages (identity table from page0), the shared caches
struct View {
    G6Attn a{};
    int* table = nullptr;
};
static View view(int pages, int page0, int pool_pages, void* index_cache, void* index_raw, unsigned* err,
                 const void* ape, const void* norm, const void* bias) {
    View v;
    std::vector<int> t(pages);
    for (int i = 0; i < pages; ++i) t[i] = page0 + i;
    v.table = dalloc<int>(pages); h2d(v.table, t);
    G6Attn& a = v.a;
    a.heads = 16; a.page_size = g6kv::PAGE; a.page_count = pool_pages; a.table_size = pages; a.query_tile = 32;
    a.context = pages * g6kv::PAGE; a.block_table = v.table; a.index_cache = index_cache; a.index_raw = index_raw;
    a.error = err; a.index_ape = ape; a.index_norm = norm; a.index_bias = bias;
    return v;
}

// rows [p, p + n) of session s with the raw rows from `k` / `g` (host [pos][128] FP16)
static void call(View& v, int s, int p, int n, const std::vector<uint16_t>& k, const std::vector<uint16_t>& g) {
    std::vector<int> pos(n), ss(n, s);
    for (int i = 0; i < n; ++i) pos[i] = p + i;
    int* dpos = dalloc<int>(n); int* dss = dalloc<int>(n);
    __half* dk = dalloc<__half>((size_t)n * 128); __half* dg = dalloc<__half>((size_t)n * 128);
    h2d(dpos, pos); h2d(dss, ss);
    CK(cudaMemcpy(dk, k.data() + (size_t)p * 128, (size_t)n * 256, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, g.data() + (size_t)p * 128, (size_t)n * 256, cudaMemcpyHostToDevice));
    G6Attn a = v.a;
    a.rows = n; a.positions = dpos; a.raw_sess = dss; a.index_k = dk; a.index_gate = dg;
    fa::kp_pool<<<n, fa::IDIM>>>(a);
    fa::kp_ring<<<std::min(n, g6kv::RAW_RING), 2 * fa::IDIM>>>(a);
    CK(cudaDeviceSynchronize());
    cudaFree(dpos); cudaFree(dss); cudaFree(dk); cudaFree(dg);
}

static void test_pools() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int N = 9000, pages = (N + 63) / 64;
    std::vector<uint16_t> ape(4 * 128), norm(128), bias(128);
    for (auto& x : ape) x = f2bf(0.3f * nd(rng));
    for (auto& x : norm) x = f2bf(1.f + 0.1f * nd(rng));
    for (auto& x : bias) x = f2bf(0.05f * nd(rng));
    uint16_t *dape = dalloc<uint16_t>(ape.size()), *dnorm = dalloc<uint16_t>(128), *dbias = dalloc<uint16_t>(128);
    h2d(dape, ape); h2d(dnorm, norm); h2d(dbias, bias);
    unsigned* err = dalloc<unsigned>(1);
    // two sessions: pages [0, pages) and [pages, 2 pages) of one pool; the reference: one call per session
    uint8_t* ref = dalloc<uint8_t>((size_t)2 * pages * g6kv::IDX_PAGE);
    uint8_t* got = dalloc<uint8_t>((size_t)2 * pages * g6kv::IDX_PAGE);
    uint8_t* ring_ref = dalloc<uint8_t>(g6kv::RAW_BYTES);
    uint8_t* ring = dalloc<uint8_t>(g6kv::RAW_BYTES);
    std::vector<uint16_t> k[2], g[2], k2[2], g2[2];
    for (int s = 0; s < 2; ++s) {
        k[s].resize((size_t)N * 128); g[s].resize((size_t)N * 128);
        for (auto& x : k[s]) x = f2h(nd(rng));
        for (auto& x : g[s]) x = f2h(nd(rng));
        k2[s] = k[s]; g2[s] = g[s];
    }
    View r[2], v[2];
    for (int s = 0; s < 2; ++s) {
        r[s] = view(pages, s * pages, 2 * pages, ref, ring_ref, err, dape, dnorm, dbias);
        v[s] = view(pages, s * pages, 2 * pages, got, ring, err, dape, dnorm, dbias);
    }
    // the chunked run: random call sizes, prefill quanta, and rollbacks that first write rows past the commit point
    // with other values (k2 / g2: garbage), then write them again with the final ones
    int at[2] = {0, 0};
    std::uniform_int_distribution<int> sz(1, 32), coin(0, 5);
    while (at[0] < N || at[1] < N) {
        const int s = (at[0] < N && (at[1] >= N || coin(rng) & 1)) ? 0 : 1;
        const int c = coin(rng);
        int n = c == 0 ? std::min(4096, N - at[s]) : std::min(sz(rng), N - at[s]);
        if (c == 1 && at[s] + 8 <= N) {   // a speculative round: 8 rows of other values, of which j are kept
            const int j = std::uniform_int_distribution<int>(0, 7)(rng);
            for (size_t i = (size_t)at[s] * 128; i < (size_t)(at[s] + 8) * 128; ++i) { k2[s][i] = f2h(nd(rng)); g2[s][i] = f2h(nd(rng)); }
            for (size_t i = (size_t)at[s] * 128; i < (size_t)(at[s] + j) * 128; ++i) { k2[s][i] = k[s][i]; g2[s][i] = g[s][i]; }
            call(v[s], s, at[s], 8, k2[s], g2[s]);
            at[s] += j;
            continue;
        }
        call(v[s], s, at[s], n, k[s], g[s]);
        at[s] += n;
    }
    for (int s = 0; s < 2; ++s) call(r[s], s, 0, N, k[s], g[s]);
    const auto a = d2h(ref, (size_t)2 * pages * g6kv::IDX_PAGE), b = d2h(got, (size_t)2 * pages * g6kv::IDX_PAGE);
    report(a == b, "pools: chunked calls with rollbacks and two sessions = one call (all pooled keys, byte for byte)");
    report(d2h(err, 1)[0] == 0, "pools: no attention error");
    // a session restarted mid-pool from its ring alone (a checkpoint restore): the ring's last rows feed the next pool
    CK(cudaMemset(got, 0, (size_t)2 * pages * g6kv::IDX_PAGE));
    call(v[0], 0, 0, 4098, k[0], g[0]);   // ends inside a pool: rows 4096, 4097 only in the ring
    call(v[0], 0, 4098, N - 4098, k[0], g[0]);
    const auto c2 = d2h(got, (size_t)pages * g6kv::IDX_PAGE);
    report(std::equal(c2.begin(), c2.end(), a.begin()), "pools: a call that starts inside a pool reads the ring");
}

// the decode scorer and kp_select over every pool vs key tiles of KP_TILE
static void test_tiles() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int pools = 5 * fa::KP_TILE + 200, ctx = pools * 4 + 3, pages = (ctx + 63) / 64, R = 48;
    const int pstride = ((ctx + 3) / 4 + 31) / 32 * 32;
    // few distinct pooled keys: many equal scores
    std::vector<float> keys((size_t)pages * 16 * 128);
    std::vector<std::vector<float>> distinct(37, std::vector<float>(128));
    for (auto& d : distinct) for (auto& x : d) x = nd(rng);
    std::uniform_int_distribution<int> pick(0, 36);
    for (size_t p = 0; p < (size_t)pages * 16; ++p) { const auto& d = distinct[pick(rng)]; std::copy(d.begin(), d.end(), keys.begin() + p * 128); }
    float* dkeys = dalloc<float>(keys.size()); h2d(dkeys, keys);
    std::vector<uint16_t> q((size_t)R * 32 * 128), w((size_t)R * 32);
    for (auto& x : q) x = f2h(nd(rng));
    for (auto& x : w) x = f2h(nd(rng));
    // rows: positions spread over the context, some exactly at tile edges, a few at <= 512 pools
    std::vector<int> pos(R);
    for (int r = 0; r < R; ++r) pos[r] = std::uniform_int_distribution<int>(0, ctx - 1)(rng);
    pos[0] = 3; pos[1] = 2047; pos[2] = 2051; pos[3] = fa::KP_TILE * 4 - 1; pos[4] = fa::KP_TILE * 4 + 3; pos[5] = ctx - 1;
    pos[6] = 2 * fa::KP_TILE * 4 + 1; pos[7] = fa::KP_TILE * 4 + 2047;
    std::sort(pos.begin(), pos.end());   // the prefill scorer's rows ascend (its last row bounds the CTA)
    int* dpos = dalloc<int>(R); h2d(dpos, pos);
    __half *dq = dalloc<__half>(q.size()), *dw = dalloc<__half>(w.size());
    CK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    unsigned* err = dalloc<unsigned>(1);
    std::vector<int> t(pages); for (int i = 0; i < pages; ++i) t[i] = i;
    int* table = dalloc<int>(pages); h2d(table, t);
    int *ia = dalloc<int>((size_t)R * fa::TOP), *ib = dalloc<int>((size_t)R * fa::TOP);
    G6Attn a{};
    a.rows = R; a.heads = 16; a.context = ctx; a.page_size = 64; a.page_count = pages; a.table_size = pages; a.query_tile = 64;
    a.positions = dpos; a.block_table = table; a.index_cache = dkeys; a.index_q = dq; a.index_weight = dw; a.error = err;
    float* full = dalloc<float>((size_t)R * pstride);
    float* tile = dalloc<float>((size_t)R * fa::KP_TILE);
    float* cs[2] = {dalloc<float>((size_t)R * 512), dalloc<float>((size_t)R * 512)};
    int* ci[2] = {dalloc<int>((size_t)R * 512), dalloc<int>((size_t)R * 512)};
    for (int pf = 0; pf < 2; ++pf) {
        CK(cudaMemset(full, 0, (size_t)R * pstride * 4));
        a.indices = ia;
        if (pf) fa_pf::kp_scores_tc<2><<<dim3(pstride / 32, (R + 63) / 64), 128>>>(a, 0, R, full, pstride, 0);
        else fa::kp_scores<<<dim3(pstride / 32, R), 256>>>(a, 0, full, pstride);
        fa::kp_select<<<R, fa::SELT>>>(a, 0, full, pstride);
        a.indices = ib;
        bool same_scores = true;
        for (int t0 = 0, k = 0; t0 < pstride; t0 += fa::KP_TILE, k ^= 1) {
            const int nt = std::min(fa::KP_TILE, pstride - t0);
            CK(cudaMemset(tile, 0, (size_t)R * fa::KP_TILE * 4));
            if (pf) fa_pf::kp_scores_tc<2><<<dim3(nt / 32, (R + 63) / 64), 128>>>(a, 0, R, tile, fa::KP_TILE, t0);
            else fa::kp_scores<<<dim3(nt / 32, R), 256>>>(a, 0, tile, fa::KP_TILE, t0);
            CK(cudaDeviceSynchronize());
            // every pool a row scores, the same bits as the one-pass score
            const auto ft = d2h(tile, (size_t)R * fa::KP_TILE), ff = d2h(full, (size_t)R * pstride);
            for (int r = 0; r < R; ++r) {
                const int nv = (pos[r] + 1) / 4;
                if (pf && nv <= 512) continue;   // the prefill scorer skips rows that take every pool
                for (int p = t0; p < std::min(t0 + nt, nv); ++p)
                    if (memcmp(&ft[(size_t)r * fa::KP_TILE + p - t0], &ff[(size_t)r * pstride + p], 4)) same_scores = false;
            }
            fa::kp_select_tile<<<R, fa::SELT>>>(a, 0, tile, fa::KP_TILE, cs[k], ci[k], cs[k ^ 1], ci[k ^ 1], t0, nt, t0 + nt >= pstride);
        }
        CK(cudaDeviceSynchronize());
        report(same_scores, pf ? "tiles: kp_scores_tc with a pool offset = one pass" : "tiles: kp_scores with a pool offset = one pass");
        const auto A = d2h(ia, (size_t)R * fa::TOP), B = d2h(ib, (size_t)R * fa::TOP);
        int bad = 0;
        for (int r = 0; r < R; ++r) {
            if (!memcmp(&A[(size_t)r * fa::TOP], &B[(size_t)r * fa::TOP], fa::TOP * 4)) continue;
            int j = 0;
            while (A[(size_t)r * fa::TOP + j] == B[(size_t)r * fa::TOP + j]) ++j;
            if (bad++ < 4) printf("  row %d (position %d): index %d is %d one pass, %d tiled\n", r, pos[r], j,
                                  A[(size_t)r * fa::TOP + j], B[(size_t)r * fa::TOP + j]);
        }
        char m[160];
        snprintf(m, sizeof m, "tiles (%s scorer): kp_select_tile = kp_select on %d rows, %d tiles, ties (%d rows differ)",
                 pf ? "prefill" : "decode", R, (pstride + fa::KP_TILE - 1) / fa::KP_TILE, bad);
        report(!bad, m);
    }
    report(d2h(err, 1)[0] == 0, "tiles: no attention error");
}

int main() {
    printf("KP_TILE %d pools, raw ring %d positions x %d sessions\n", fa::KP_TILE, g6kv::RAW_RING, g6kv::RAW_SESS);
    test_pools();
    test_tiles();
    printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
