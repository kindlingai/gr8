// The Flash kpool indexer's key tiles (attn/kpool.cuh): on random pooled keys with many equal scores, kp_select_tile
// over key tiles of KP_TILE pools (built small here) gives kp_select's lists, ties included, after the decode scorer
// (kp_scores) and the prefill tensor-core scorer (kp_scores_tc), whose per-pool scores with a pool offset equal the
// one-pass scores. Rows sit below, at and past the tile edges, and some see at most 512 pools.
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

// the decode scorer and kp_select over every pool vs key tiles of KP_TILE
static void test_tiles() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int pools = 5 * fa::KP_TILE + 200, ctx = pools * 4 + 3, pages = (ctx + 63) / 64, R = 48;
    const int pstride = ((ctx + 3) / 4 + 31) / 32 * 32;
    // few distinct pooled keys: many equal scores
    // the index cache: page p's 16 pooled keys at p * IDX_PAGE + IDX_POOL_OFF (attn/kvfmt.h)
    constexpr size_t PW = g6kv::IDX_PAGE / 4, OW = g6kv::IDX_POOL_OFF / 4;
    std::vector<float> keys((size_t)pages * PW);
    std::vector<std::vector<float>> distinct(37, std::vector<float>(128));
    for (auto& d : distinct) for (auto& x : d) x = nd(rng);
    std::uniform_int_distribution<int> pick(0, 36);
    for (size_t p = 0; p < (size_t)pages * 16; ++p) {
        const auto& d = distinct[pick(rng)];
        std::copy(d.begin(), d.end(), keys.begin() + (p / 16) * PW + OW + (p % 16) * 128);
    }
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
    printf("KP_TILE %d pools\n", fa::KP_TILE);
    test_tiles();
    printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
