// DS41 indexer key tiles (ds41_idx.cuh stage_index_tiled): the tiled prefill scratch gives the same top-512 lists and the same layer-20
// top-2048 candidate blocks as the one-pass scratch, on random data, for ratio-1 and ratio-2 key counts below, at and above one tile and at a
// long position, for every dense logits kernel and both top-k kernels. Also times one 4096-row chunk (one pass vs tiles).
// Build & run on a Spark (sm_121a; RTX 5090: compute_120a / sm_120a):
//   nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -I engine engine/tests/test_ds41_idx_tile.cu -o /tmp/test_ds41_idx_tile
//   /tmp/test_ds41_idx_tile            equality cases (< 1 GiB of device memory)
//   /tmp/test_ds41_idx_tile --long     + 262,144 / 524,288 keys
//   /tmp/test_ds41_idx_tile --time     + the 4096-row chunk timing at 32K and 256K keys (about 2.6 GiB for the one-pass logits)
#include "../attn/ds41_idx.cuh"
#include "../attn/ds41_idx_layout.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ds41i;

#define CK(x)                                                                                                  \
    do {                                                                                                       \
        cudaError_t e_ = (x);                                                                                  \
        if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } \
    } while (0)

static int g_fail = 0;
template <class T> static T* dalloc(size_t n) { T* p = nullptr; CK(cudaMalloc(&p, n * sizeof(T) + 256)); CK(cudaMemset(p, 0, n * sizeof(T) + 256)); return p; }

struct Case
{
    const char* name;
    int R, ratio, maxpos;   // R query rows, key ratio, highest query position + 1
    int tile;               // key tile of the tiled carve
    int ties;               // 1: few distinct key rows (many equal logits)
    int logits;             // 0 dp4a, 1 mma exact, 2 mma fast, 3 pipe_b, 4 pipe_qr
    int mc;                 // multi-CTA top-k
};

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16); }

// one case: returns 0 = identical
static int run(const Case& c, bool timing = false, float* ms_flat = nullptr, float* ms_tiled = nullptr)
{
    std::mt19937 rng(1234u + c.R * 7 + c.maxpos * 3 + c.ratio + c.ties * 11 + c.logits * 13);
    const int R = c.R;
    const int N = c.maxpos / c.ratio;   // largest klen
    const int rpp = ica_aligned_rows(N + 64);
    // q_b, wproj
    std::vector<uint16_t> hq((size_t)R * IH * ID), hw((size_t)R * IH);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hq) v = f2bf(nd(rng));
    for (auto& v : hw) v = f2bf(nd(rng));
    // positions: row 0 at maxpos - 1, the rest random (some short: rows with fewer than 512 keys, inactive in later tiles)
    std::vector<int> hpos(R);
    for (int r = 0; r < R; r++)
    {
        const int sel = rng() % 8;
        hpos[r] = r == 0 ? c.maxpos - 1 : sel == 0 ? (int)(rng() % std::min(c.maxpos, 900 * c.ratio)) : (int)(rng() % c.maxpos);
    }
    // ica pool: two slots (the rows sit in slot 1), planar codes then scale words
    const size_t page = (size_t)rpp * (IK_CODES + IK_SF);
    std::vector<uint8_t> hk(2 * page, 0);
    uint8_t* pg = hk.data() + page;
    const int distinct = c.ties ? 5 : rpp;
    for (int j = 0; j < rpp; j++)
    {
        const int src = c.ties ? (int)(rng() % distinct) : j;
        std::mt19937 rj(77u + src);
        for (int b = 0; b < IK_CODES; b++) pg[(size_t)j * IK_CODES + b] = (uint8_t)rj();
        for (int g = 0; g < IK_SF; g++) pg[(size_t)rpp * IK_CODES + (size_t)j * IK_SF + g] = (uint8_t)(124 + rj() % 6);
    }
    bf16* q_b = dalloc<bf16>(hq.size());
    bf16* wp = dalloc<bf16>(hw.size());
    int* dpos = dalloc<int>(R);
    int32_t* klen = dalloc<int32_t>(R);
    uint8_t* ica = dalloc<uint8_t>(hk.size());
    int32_t* pages = dalloc<int32_t>(1);
    const int one = 1;
    CK(cudaMemcpy(q_b, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(wp, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos, hpos.data(), R * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ica, hk.data(), hk.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(pages, &one, 4, cudaMemcpyHostToDevice));
    launch_klen(dpos, c.ratio, klen, R, 0);

    IndexIn in{};
    in.q_b = q_b; in.wproj = wp; in.inv = rope_inv_ds41(); in.pos0 = 0; in.R = R; in.klen = klen; in.max_klen = N;
    in.k = ica_map(ica, pages, rpp, page); in.dpos = dpos;
    in.logits_mma = c.logits == 1 ? 1 : c.logits == 2 ? 2 : 0;
    in.pipe_b = c.logits == 3;
    in.pipe_qr = c.logits == 4;
    in.topk_mc = c.mc;
    // the carve basis (carve_rows: chunk_end, ratio-1 rows). Equality cases: at least one tile past TOPK + tile, so the scratch is the tiled
    // one even when this call's keys fit in one tile (the decode-graph basis is the capacity, longer than the keys). Timing: the chunk's own.
    const int max_rows = timing ? c.maxpos : std::max(c.maxpos, TOPK + c.tile + 1);
    const size_t b_flat = Scratch::bytes(R, max_rows, 0), b_tile = Scratch::bytes(R, max_rows, c.tile);
    void* s_flat = dalloc<uint8_t>(b_flat);
    void* s_tile = dalloc<uint8_t>(b_tile);
    const Scratch A = Scratch::carve(s_flat, R, max_rows, R, 0);
    const Scratch B = Scratch::carve(s_tile, R, max_rows, R, c.tile);
    int32_t* oa = dalloc<int32_t>((size_t)R * TOPK);
    int32_t* ob = dalloc<int32_t>((size_t)R * TOPK);
    int32_t* ob2 = dalloc<int32_t>((size_t)R * CAND_BLOCKS);
    const bool cand = c.ratio == 1;   // layer 20 (ratio 1) also makes the candidate blocks
    auto flat = [&] { stage_index_dense(in, A, oa, TOPK, 0); if (cand) stage_cand_blocks(in, A, 0); };
    auto tiled = [&] { stage_index_dense(in, B, ob, TOPK, 0, cand); if (cand) stage_cand_blocks(in, B, 0, true); };
    flat();
    tiled();
    CK(cudaDeviceSynchronize());
    // the recompute path: cand_blocks without cand_src (its own tiled pass), into a copy of the list
    if (cand)
    {
        CK(cudaMemcpy(ob2, B.cand, (size_t)R * CAND_BLOCKS * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemset(B.cand, 0x7F, (size_t)R * CAND_BLOCKS * 4));
        stage_cand_blocks(in, B, 0, false);
        CK(cudaDeviceSynchronize());
    }
    std::vector<int32_t> ha((size_t)R * TOPK), hb(ha.size()), ca((size_t)R * CAND_BLOCKS), cb(ca.size()), cb2(ca.size());
    CK(cudaMemcpy(ha.data(), oa, ha.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), ob, hb.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0, bad_c = 0, bad_c2 = 0, rows_short = 0, ties_cut = 0;
    for (size_t i = 0; i < ha.size(); i++) bad += ha[i] != hb[i];
    if (cand)
    {
        CK(cudaMemcpy(ca.data(), A.cand, ca.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cb.data(), B.cand, cb.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cb2.data(), ob2, cb2.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < ca.size(); i++) { bad_c += ca[i] != cb2[i]; bad_c2 += ca[i] != cb[i]; }
    }
    // sanity of the reference: sorted ascending, -1 tail, rows with fewer than 512 keys present; ties at the cut (equal logits) present
    {
        std::vector<uint16_t> pl((size_t)R * A.ldp);
        CK(cudaMemcpy(pl.data(), A.p, pl.size() * 2, cudaMemcpyDeviceToHost));   // the flat path's logits (cand_blocks leaves p as is)
        for (int r = 0; r < R; r++)
        {
            const int kl = (hpos[r] + 1) / c.ratio;
            rows_short += kl < TOPK;
            const int32_t* o = &ha[(size_t)r * TOPK];
            for (int j = 1; j < TOPK; j++)
                if (o[j] >= 0 && o[j] <= o[j - 1]) { fprintf(stderr, "  reference row %d not ascending\n", r); g_fail++; break; }
            if (kl > TOPK)
            {
                // the cut value is the min selected logit; a tie at the cut = an unselected key with the same bits
                uint16_t cut = 0; uint32_t cutk = 0xFFFFFFFFu;
                auto okh = [](uint16_t b) -> uint32_t { if ((b & 0x7FFFu) > 0x7F80u) return 0xFFFFu; return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u); };
                std::vector<char> sel(kl, 0);
                for (int j = 0; j < TOPK; j++) { const uint16_t b = pl[(size_t)r * A.ldp + o[j]]; sel[o[j]] = 1; if (okh(b) < cutk) { cutk = okh(b); cut = b; } }
                for (int t = 0; t < kl; t++) if (!sel[t] && pl[(size_t)r * A.ldp + t] == cut) { ties_cut++; break; }
            }
        }
    }
    float tf = 0, tt = 0;
    if (timing)
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        auto tm = [&](auto&& f) {
            f(); CK(cudaDeviceSynchronize());
            const int it = 3;
            CK(cudaEventRecord(e0));
            for (int i = 0; i < it; i++) f();
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            return ms / it;
        };
        tf = tm(flat); tt = tm(tiled);
        if (ms_flat) *ms_flat = tf;
        if (ms_tiled) *ms_tiled = tt;
    }
    const bool ok = bad == 0 && bad_c == 0 && bad_c2 == 0;
    printf("%-28s R %4d ratio %d keys %7d tile %5d (%d tiles, mode %s) logits %d mc %d: top512 diff %d, cand diff %d / fused %d; "
           "rows<512 keys %d, rows with a tie at the cut %d; scratch %.1f -> %.1f MiB%s%s\n",
           c.name, R, c.ratio, N, c.tile, B.tile ? (N + B.tile - 1) / B.tile : 1, B.tile ? "tiled" : "ONE PASS", c.logits, c.mc, bad,
           bad_c, bad_c2, rows_short, ties_cut, b_flat / 1048576.0, b_tile / 1048576.0, ok ? " OK" : " FAIL",
           timing ? (" | time one pass " + std::to_string(tf) + " ms, tiled " + std::to_string(tt) + " ms").c_str() : "");
    if (!ok) g_fail++;
    if (!timing && !B.tile && c.tile) { fprintf(stderr, "  case %s did not tile\n", c.name); g_fail++; }
    cudaFree(q_b); cudaFree(wp); cudaFree(dpos); cudaFree(klen); cudaFree(ica); cudaFree(pages); cudaFree(s_flat); cudaFree(s_tile);
    cudaFree(oa); cudaFree(ob); cudaFree(ob2);
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    bool lng = false, tim = false;
    for (int i = 1; i < argc; i++) { lng |= !strcmp(argv[i], "--long"); tim |= !strcmp(argv[i], "--time"); }
    {
        int limit = 0, dev = 0;
        CK(cudaGetDevice(&dev));
        CK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        if ((size_t)limit >= PIPE_QR_SMEM_BYTES) CK(cudaFuncSetAttribute(k_logits_pipe_qr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PIPE_QR_SMEM_BYTES));
    }
    const int T = 16384, T2 = 32768;
    // keys below a tile (one tile, still the tiled carve: carry-less first tile only), at a tile, one key over, several tiles
    std::vector<Case> cs = {
        {"r1 below tile", 96, 1, T - 1000, T, 0, 1, 1},   // the carve basis is past TOPK + T (tiled scratch); keys fit one tile
        {"r1 = tile", 96, 1, T, T, 0, 1, 1},
        {"r1 tile+1", 96, 1, T + 1, T, 0, 1, 1},
        {"r1 3.4 tiles", 64, 1, 3 * T + 7000, T, 0, 1, 1},
        {"r2 below tile", 96, 2, 2 * (T - 3000), T, 0, 1, 1},
        {"r2 = tile", 96, 2, 2 * T, T, 0, 1, 1},
        {"r2 tile+1", 96, 2, 2 * (T + 1) + 1, T, 0, 1, 1},
        {"r2 2.6 tiles", 64, 2, 2 * (2 * T + 10000), T, 0, 1, 1},
        {"r1 ties", 64, 1, 2 * T + 5000, T, 1, 1, 1},
        {"r2 ties", 64, 2, 2 * (2 * T + 5000), T, 1, 1, 1},
        {"r1 ties 1cta", 32, 1, 2 * T + 5000, T, 1, 1, 0},
        {"r1 dp4a", 32, 1, 2 * T + 3000, T, 0, 0, 1},
        {"r1 mmafast", 32, 1, 2 * T + 3000, T, 0, 2, 1},
        {"r1 pipe_b", 32, 1, 2 * T + 3000, T, 0, 3, 1},
        {"r1 pipe_qr", 32, 1, 2 * T + 3000, T, 0, 4, 1},
        {"r2 pipe_b ties", 32, 2, 2 * (2 * T + 3000), T, 1, 3, 1},
        {"r1 tile 32K", 64, 1, 3 * T2 + 999, T2, 0, 1, 1},
        {"r1 1cta", 32, 1, 3 * T + 999, T, 0, 1, 0},
    };
    if (lng)
    {
        cs.push_back({"r1 long 262144", 64, 1, 262144, T2, 0, 1, 1});
        cs.push_back({"r2 long 262144 keys", 64, 2, 2 * 262144, T2, 0, 1, 1});
        cs.push_back({"r1 long 262144 ties", 32, 1, 262144 + 77, T2, 1, 1, 1});
        cs.push_back({"r1 long 524288", 32, 1, 524288, T2, 0, 1, 1});
    }
    for (const Case& c : cs) run(c);
    if (tim)
    {
        for (int keys : {32768, 262144})
            for (int t : {16384, 32768, 65536})
            {
                Case c{"time 4096 rows", 4096, 1, keys, t, 0, 1, 1};
                run(c, true);
            }
    }
    printf(g_fail ? "FAIL (%d)\n" : "ALL OK\n", g_fail);
    return g_fail ? 1 : 0;
}
