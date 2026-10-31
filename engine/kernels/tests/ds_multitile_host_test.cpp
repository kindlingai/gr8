// CPU test of the multi-tile MoE plan and tiling logic (ds_plan_mt.h), M = 1..32. No GPU, no CUDA headers.
//
//   g++ -O2 -std=c++17 -Wall -ffp-contract=off -o ds_multitile_host_test ds_multitile_host_test.cpp && ./ds_multitile_host_test [p128_dir]
//   p128_dir: reference-model route dumps of one 128-token prompt (L<n>.router_idx.bin int32 [128][6], layers 0/2/20/39).
//   Without it the real-route cases are skipped.
//
// What it checks:
//   T1  mt_frag_decode: a bijection over the tile tables; tile 0 is exactly mx4::build_xfrag's index arithmetic.
//   T2  plan_ref invariants on uniform / clustered / identical / one-hot-expert / real (p128) routes, M = 1..32, (NE,TK) = (384,6), (128,3);
//       and the kernel's index helpers (mt_ntiles, mt_tile_mask, mt_route_pos) agree with plan_ref's independent loops.
//   T3  fc1's B-fragment gather: for every (slot tile, lane, k16 step, reg) the word read at mt_xoff(...) is the bf16/fp16 pair of the token that
//       owns the lane's column (the m16n8k16 B layout: lane l = column l>>2, k pair 2 (l&3), +8 for reg 1).
//   T4  the whole pipeline's index chain (slot -> tile -> column -> act -> ypart -> combine) on small dense experts in double, bit-equal to the
//       naive per-token MoE.
//   T5  scratch bounds: n_slots, n_tiles <= MM*TK in every case; the worst case reaches them.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <set>
#include "../ds_plan_mt.h"
using namespace dsr;

static int g_fail = 0;
#define EXPECT(c, ...) do { if (!(c)) { if (g_fail < 40) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } ++g_fail; } } while (0)

// ------------------------------------------------------------------------------------------------ route generators
enum Kind { UNIFORM, CLUSTER, IDENTICAL, HOT, DISTINCT, KINDS };
static const char* kname[KINDS] = {"uniform", "cluster", "identical", "hot-expert", "all-distinct"};

static void gen(Kind kind, std::mt19937_64& g, int M, int NE, int TK, int (*ex)[8], float (*w)[8])
{
    std::vector<int> pool(NE);
    for (int i = 0; i < NE; ++i) pool[i] = i;
    std::shuffle(pool.begin(), pool.end(), g);
    const int P = kind == CLUSTER ? TK + 2 + static_cast<int>(g() % 6) : kind == IDENTICAL ? TK : NE;     // experts the tokens draw from
    const int hot = pool[0];
    for (int m = 0; m < M; ++m)
    {
        std::vector<int> cand(pool.begin(), pool.begin() + P);
        if (kind == DISTINCT)
        {
            for (int k = 0; k < TK; ++k) ex[m][k] = pool[(m * TK + k) % NE];       // NE >= M*TK for 384/6/M<=32 only: otherwise repeats across tokens (fine)
        }
        else
        {
            if (kind == HOT) cand.erase(std::remove(cand.begin(), cand.end(), hot), cand.end());
            std::shuffle(cand.begin(), cand.end(), g);
            int k = 0;
            if (kind == HOT) ex[m][k++] = hot;
            for (; k < TK; ++k) ex[m][k] = cand[k - (kind == HOT ? 1 : 0)];
            if (kind == IDENTICAL) std::shuffle(ex[m], ex[m] + TK, g);             // same set, different selection order
        }
        double s = 0;
        for (int k = 0; k < TK; ++k) { w[m][k] = 0.1f + static_cast<float>(g() % 1000) / 1000.f; s += w[m][k]; }
        for (int k = 0; k < TK; ++k) w[m][k] = static_cast<float>(w[m][k] / s * 1.5);
        for (int k = TK; k < 8; ++k) { ex[m][k] = -1; w[m][k] = 0.f; }
    }
}

// ------------------------------------------------------------------------------------------------ T2
template <int TK, int MM>
static void check_plan(const int (*ex)[8], const float (*w)[8], int M, int NE, const char* what, int* max_slots = nullptr, int* max_tiles = nullptr)
{
    static PlanMT<TK, MM> P;
    constexpr int SM = PlanMT<TK, MM>::SM;
    plan_ref<TK, MM>(M, ex, w, P);
    std::set<int> distinct;
    std::vector<int> cnt(NE, 0);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < TK; ++k) { distinct.insert(ex[m][k]); ++cnt[ex[m][k]]; }
    EXPECT(P.n_slots == static_cast<int>(distinct.size()), "%s: n_slots %d != distinct %zu", what, P.n_slots, distinct.size());
    EXPECT(P.n_slots <= SM && P.n_tiles <= SM && P.n_slots <= P.n_tiles, "%s: n_slots %d n_tiles %d exceed SM %d", what, P.n_slots, P.n_tiles, SM);
    int tiles = 0, prev = -1;
    for (int s = 0; s < SM; ++s)
    {
        if (s >= P.n_slots) { EXPECT(P.slot_expert[s] == -1 && P.slot_nt[s] == 0, "%s: slot %d beyond n_slots not empty", what, s); continue; }
        const int e = P.slot_expert[s], n = cnt[e];
        EXPECT(e > prev, "%s: slot_expert not ascending at %d", what, s);
        prev = e;
        EXPECT(P.slot_tile0[s] == tiles, "%s: slot %d tile0 %d != running %d", what, s, P.slot_tile0[s], tiles);
        EXPECT(P.slot_nt[s] == mt_ntiles(n) && P.slot_nt[s] >= 1 && P.slot_nt[s] <= MM / 8, "%s: slot %d nt %d for n %d", what, s, P.slot_nt[s], n);
        int toks_seen = 0, last_tok = -1;
        for (int j = 0; j < P.slot_nt[s]; ++j)
        {
            const int t = P.slot_tile0[s] + j;
            EXPECT(P.tile_mask[t] == mt_tile_mask(n, j), "%s: tile %d mask %x != helper %x", what, t, P.tile_mask[t], mt_tile_mask(n, j));
            const int live = mt_popc(static_cast<unsigned>(P.tile_mask[t]));
            EXPECT((P.tile_mask[t] & (P.tile_mask[t] + 1)) == 0, "%s: tile %d mask %x is not the low bits", what, t, P.tile_mask[t]);
            EXPECT(j + 1 == P.slot_nt[s] || live == 8, "%s: inner tile %d not full", what, t);
            for (int c = 0; c < 8; ++c)
            {
                const bool is = (P.tile_mask[t] >> c) & 1;
                EXPECT((P.tile_tok[t][c] >= 0) == is, "%s: tile %d col %d tok/mask mismatch", what, t, c);
                if (!is) { EXPECT(P.tile_w[t][c] == 0.f, "%s: unused col has weight", what); continue; }
                const int m = P.tile_tok[t][c];
                EXPECT(m > last_tok, "%s: tokens not ascending in slot %d", what, s);
                last_tok = m;
                int kk = -1;
                for (int k = 0; k < TK; ++k) if (ex[m][k] == e) kk = k;
                EXPECT(kk >= 0, "%s: tile %d col %d holds token %d which does not route to expert %d", what, t, c, m, e);
                if (kk < 0) continue;
                EXPECT(P.tile_w[t][c] == w[m][kk], "%s: tile weight", what);
                EXPECT(P.tok_slot[m][kk] == t && P.tok_col[m][kk] == c, "%s: token %d route %d -> (%d,%d), plan says (%d,%d)", what, m, kk, P.tok_slot[m][kk], P.tok_col[m][kk], t, c);
                // the kernel's formula: tile0 from the scan, rank from the expert's token mask
                unsigned em = 0;
                for (int m2 = 0; m2 < M; ++m2) for (int k2 = 0; k2 < TK; ++k2) if (ex[m2][k2] == e) em |= 1u << m2;
                int ht, hc;
                mt_route_pos(P.slot_tile0[s], em, m, ht, hc);
                EXPECT(ht == t && hc == c, "%s: mt_route_pos(%d,%x,%d) = (%d,%d), spec (%d,%d)", what, P.slot_tile0[s], em, m, ht, hc, t, c);
                ++toks_seen;
            }
        }
        EXPECT(toks_seen == n, "%s: slot %d holds %d of %d tokens", what, s, toks_seen, n);
        tiles += P.slot_nt[s];
    }
    EXPECT(tiles == P.n_tiles, "%s: n_tiles %d != sum %d", what, P.n_tiles, tiles);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < TK; ++k)
        {
            const int t = P.tok_slot[m][k];
            EXPECT(t >= 0 && t < P.n_tiles && P.tile_tok[t][P.tok_col[m][k]] == m && P.tok_expert[m][k] == ex[m][k], "%s: route (%d,%d) not placed", what, m, k);
        }
    for (int m = M; m < MM; ++m) for (int k = 0; k < 8; ++k) EXPECT(P.tok_slot[m][k] == -1, "%s: row %d beyond M has a route", what, m);
    for (int m = 0; m < M; ++m) for (int k = TK; k < 8; ++k) EXPECT(P.tok_slot[m][k] == -1, "%s: route %d >= TK placed", what, k);
    if (max_slots) *max_slots = std::max(*max_slots, P.n_slots);
    if (max_tiles) *max_tiles = std::max(*max_tiles, P.n_tiles);
}

// ------------------------------------------------------------------------------------------------ T1
static void t1_frag()
{
    for (int K1 : {64, 5120})
    {
        const int NT = 4, n = NT * K1 * 4;
        std::vector<char> seen(n, 0);
        for (int tile = 0; tile < NT; ++tile)
            for (int ks = 0; ks < K1 / 16; ++ks)
                for (int lane = 0; lane < 32; ++lane)
                    for (int reg = 0; reg < 2; ++reg)
                    {
                        const int i = tile * K1 * 4 + ks * 64 + lane * 2 + reg;
                        const MtFrag f = mt_frag_decode(i, K1);
                        EXPECT(f.tile == tile && f.ks == ks && f.lane == lane && f.reg == reg, "decode(%d) wrong", i);
                        // mx4::build_xfrag: ks = i>>6, l = (i>>1)&31, reg = i&1, t = l>>2, element x[t*K1 + ks*16 + reg*8 + 2*(l&3)]
                        const int r = i % (K1 * 4);
                        EXPECT(f.col == ((r >> 1) & 31) >> 2 && f.k0 == (r >> 6) * 16 + (r & 1) * 8 + 2 * (((r >> 1) & 31) & 3), "decode(%d) vs build_xfrag arithmetic", i);
                        EXPECT(f.k0 + 1 < K1, "k pair out of range");
                        EXPECT(!seen[i], "word %d twice", i);
                        seen[i] = 1;
                    }
        EXPECT(std::count(seen.begin(), seen.end(), 1) == n, "bijection K1=%d", K1);
    }
    printf("T1 fragment-table index: bijection over 4 tiles, tile 0 = build_xfrag arithmetic (K1 = 64, 5120)\n");
}

// ------------------------------------------------------------------------------------------------ T3
// A "token row" is K1 pairs; the 32-bit word (k pair) is encoded as token*2^20 + k0 so a wrong gather shows which token / k it came from.
template <int TK, int MM>
static void t3_gather(std::mt19937_64& g, int K1, int M, int NE, Kind kind)
{
    static int ex[MM][8]; static float w[MM][8];
    gen(kind, g, M, NE, TK, ex, w);
    static PlanMT<TK, MM> P;
    plan_ref<TK, MM>(M, ex, w, P);
    const int ts = K1 * 4, NT = (M + 7) / 8;
    std::vector<uint32_t> table(static_cast<size_t>(NT) * ts, 0xdeadbeefu);
    for (int i = 0; i < NT * ts; ++i)               // build_xfrag_mt: word = pair of token 8 tile + col (zero past M)
    {
        const MtFrag f = mt_frag_decode(i, K1);
        const int tok = 8 * f.tile + f.col;
        table[i] = tok < M ? static_cast<uint32_t>(tok) * (1u << 20) + static_cast<uint32_t>(f.k0) : 0u;
    }
    long checked = 0;
    for (int t = 0; t < P.n_tiles; ++t)
        for (int lane = 0; lane < 32; ++lane)
        {
            const int tau = P.tile_tok[t][lane >> 2];
            const int off = mt_xoff(tau, lane, ts);
            EXPECT(off >= 0 && off + (K1 / 16 - 1) * 64 + 1 < NT * ts, "gather offset out of the table");
            const int want_tok = tau < 0 ? 0 : tau;
            for (int ks = 0; ks < K1 / 16; ++ks)
                for (int reg = 0; reg < 2; ++reg)
                {
                    const uint32_t v = table[off + ks * 64 + reg];
                    const uint32_t k0 = static_cast<uint32_t>(ks * 16 + reg * 8 + 2 * (lane & 3));     // m16n8k16 B: k = 2 (l&3) + 8 reg, n = l>>2
                    EXPECT(v == static_cast<uint32_t>(want_tok) * (1u << 20) + k0, "gather tile %d lane %d ks %d reg %d: got tok %u k %u, want tok %d k %u", t, lane, ks, reg, v >> 20, v & 0xfffff, want_tok, k0);
                    ++checked;
                }
        }
    (void)checked;
}

// ------------------------------------------------------------------------------------------------ T4
template <int TK, int MM>
static void t4_pipeline(std::mt19937_64& g, int M, int NE, Kind kind)
{
    constexpr int K = 32, I = 16, NO = 24;
    static int ex[MM][8]; static float w[MM][8];
    gen(kind, g, M, NE, TK, ex, w);
    static PlanMT<TK, MM> P;
    plan_ref<TK, MM>(M, ex, w, P);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double> x(static_cast<size_t>(M) * K), W1(static_cast<size_t>(NE) * I * K), W3(W1.size()), W2(static_cast<size_t>(NE) * NO * I), sh(static_cast<size_t>(NO));
    for (auto& v : x) v = u(g);
    for (auto& v : W1) v = u(g);
    for (auto& v : W3) v = u(g);
    for (auto& v : W2) v = u(g);
    auto act_of = [&](int e, const double* xr, double* a) {
        for (int i = 0; i < I; ++i)
        {
            double gt = 0, up = 0;
            for (int k = 0; k < K; ++k) { gt += W1[(static_cast<size_t>(e) * I + i) * K + k] * xr[k]; up += W3[(static_cast<size_t>(e) * I + i) * K + k] * xr[k]; }
            a[i] = gt / (1.0 + std::exp(-gt)) * up;
        }
    };
    auto down = [&](int e, const double* a, int o) { double y = 0; for (int i = 0; i < I; ++i) y += W2[(static_cast<size_t>(e) * NO + o) * I + i] * a[i]; return y; };
    // the mt pipeline: per slot, per tile, per column (token = tile_tok), act[tile][col][i], ypart[tile][col][o] = tile_w * down(act)
    std::vector<double> act(static_cast<size_t>(P.n_tiles) * 8 * I, std::nan("")), yp(static_cast<size_t>(P.n_tiles) * 8 * NO, std::nan(""));
    for (int s = 0; s < P.n_slots; ++s)
        for (int j = 0; j < P.slot_nt[s]; ++j)
        {
            const int t = P.slot_tile0[s] + j;
            for (int c = 0; c < 8; ++c)
            {
                if (!((P.tile_mask[t] >> c) & 1)) continue;
                act_of(P.slot_expert[s], &x[static_cast<size_t>(P.tile_tok[t][c]) * K], &act[(static_cast<size_t>(t) * 8 + c) * I]);
                for (int o = 0; o < NO; ++o) yp[(static_cast<size_t>(t) * 8 + c) * NO + o] = P.tile_w[t][c] * down(P.slot_expert[s], &act[(static_cast<size_t>(t) * 8 + c) * I], o);
            }
        }
    long bad = 0;
    for (int m = 0; m < M; ++m)
        for (int o = 0; o < NO; ++o)
        {
            double got = 0, want = 0;
            for (int k = 0; k < TK; ++k) got += yp[(static_cast<size_t>(P.tok_slot[m][k]) * 8 + P.tok_col[m][k]) * NO + o];          // k_combine's formula
            for (int k = 0; k < TK; ++k) { double a[I]; act_of(ex[m][k], &x[static_cast<size_t>(m) * K], a); want += w[m][k] * down(ex[m][k], a, o); }
            bad += !(got == want);       // exact: same terms, same order
        }
    EXPECT(bad == 0, "pipeline M=%d %s: %ld outputs differ from the naive MoE", M, kname[kind], bad);
}

// ------------------------------------------------------------------------------------------------ T6 (prefill W = 33..256: chunked plan)
// plan_ref_chunked (k_plan_pw's specification) against plan_ref: same tiles, one slot per 32-rank chunk.
template <int TK, int MM>
static void check_plan_chunked(const int (*ex)[8], const float (*w)[8], int M, int NE, const char* what, int* max_slots = nullptr, int* max_tiles = nullptr, int* max_chunks = nullptr)
{
    static PlanMT<TK, MM> C, U;
    constexpr int SM = PlanMT<TK, MM>::SM;
    constexpr int MW = MM / 32;
    plan_ref_chunked<TK, MM>(M, ex, w, C);
    plan_ref<TK, MM>(M, ex, w, U);
    // (1) the tile layout is the unchunked one, field by field
    EXPECT(C.n_tiles == U.n_tiles, "%s: n_tiles chunked %d != unchunked %d", what, C.n_tiles, U.n_tiles);
    EXPECT(std::memcmp(C.tile_mask, U.tile_mask, sizeof C.tile_mask) == 0, "%s: tile_mask differs", what);
    EXPECT(std::memcmp(C.tile_w, U.tile_w, sizeof C.tile_w) == 0, "%s: tile_w differs", what);
    EXPECT(std::memcmp(C.tile_tok, U.tile_tok, sizeof C.tile_tok) == 0, "%s: tile_tok differs", what);
    EXPECT(std::memcmp(C.tok_slot, U.tok_slot, sizeof C.tok_slot) == 0, "%s: tok_slot differs", what);
    EXPECT(std::memcmp(C.tok_col, U.tok_col, sizeof C.tok_col) == 0, "%s: tok_col differs", what);
    EXPECT(std::memcmp(C.tok_w, U.tok_w, sizeof C.tok_w) == 0, "%s: tok_w differs", what);
    EXPECT(std::memcmp(C.tok_expert, U.tok_expert, sizeof C.tok_expert) == 0, "%s: tok_expert differs", what);
    // (2) the slots are the chunk expansion of the unchunked slots
    std::vector<int> first_slot(NE, -1);
    std::vector<int> cover(SM, 0);
    int cs = 0, chunks_max = 0;
    for (int s = 0; s < U.n_slots; ++s)
    {
        const int e = U.slot_expert[s], t0 = U.slot_tile0[s], nt = U.slot_nt[s];
        int n = 0;
        for (int j = 0; j < nt; ++j) n += mt_popc(static_cast<unsigned>(U.tile_mask[t0 + j]));
        const int nch = mt_nchunks(n);
        EXPECT(nch == (nt + 3) / 4, "%s: expert %d n %d: chunks %d vs tiles %d", what, e, n, nch, nt);
        chunks_max = std::max(chunks_max, nch);
        first_slot[e] = cs;
        for (int c = 0; c < nch; ++c, ++cs)
        {
            EXPECT(cs < SM, "%s: slot index %d out of range", what, cs);
            if (cs >= SM) return;
            const int want_nt = std::min(4, nt - 4 * c), live = std::min(32, n - 32 * c);
            EXPECT(C.slot_expert[cs] == e && C.slot_tile0[cs] == t0 + 4 * c && C.slot_nt[cs] == want_nt,
                   "%s: slot %d = (e %d, t0 %d, nt %d), want (%d, %d, %d)", what, cs, C.slot_expert[cs], C.slot_tile0[cs], C.slot_nt[cs], e, t0 + 4 * c, want_nt);
            int got_live = 0, last = -1;
            for (int j = 0; j < C.slot_nt[cs]; ++j)
            {
                const int t = C.slot_tile0[cs] + j;
                ++cover[t];
                got_live += mt_popc(static_cast<unsigned>(C.tile_mask[t]));
                for (int col = 0; col < 8; ++col)
                    if ((C.tile_mask[t] >> col) & 1) { EXPECT(C.tile_tok[t][col] > last, "%s: slot %d tokens not ascending", what, cs); last = C.tile_tok[t][col]; }
            }
            EXPECT(got_live == live, "%s: slot %d holds %d tokens, chunk should hold %d", what, cs, got_live, live);
        }
    }
    EXPECT(C.n_slots == cs, "%s: n_slots %d != chunk expansion %d", what, C.n_slots, cs);
    for (int s = C.n_slots; s < SM; ++s) EXPECT(C.slot_expert[s] == -1 && C.slot_nt[s] == 0 && C.slot_tile0[s] == 0, "%s: slot %d beyond n_slots not empty", what, s);
    for (int t = 0; t < C.n_tiles; ++t) EXPECT(cover[t] == 1, "%s: tile %d covered by %d slots", what, t, cover[t]);
    // (3) scratch bounds the adapter sizes by
    const int R = M * TK;
    EXPECT(C.n_slots <= pw_slot_bound(R, NE), "%s: n_slots %d > pw_slot_bound %d", what, C.n_slots, pw_slot_bound(R, NE));
    EXPECT(C.n_tiles <= pw_tile_bound(R, NE), "%s: n_tiles %d > pw_tile_bound %d", what, C.n_tiles, pw_tile_bound(R, NE));
    EXPECT(pw_slot_bound(R, NE) <= SM && pw_tile_bound(R, NE) <= SM, "%s: bounds exceed SM", what);
    // (4) the kernel's W-word helpers against the plan: expert token mask (MW words) -> (tile, column) of every route
    std::vector<unsigned> em(static_cast<size_t>(NE) * MW, 0u);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < TK; ++k) em[static_cast<size_t>(ex[m][k]) * MW + (m >> 5)] |= 1u << (m & 31);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < TK; ++k)
        {
            const int e = ex[m][k];
            const unsigned* ew = &em[static_cast<size_t>(e) * MW];
            int rank = 0;
            for (int m2 = 0; m2 < m; ++m2) for (int k2 = 0; k2 < TK; ++k2) rank += ex[m2][k2] == e;
            EXPECT(mt_rank_w(ew, m) == rank, "%s: mt_rank_w(e %d, m %d) = %d, spec %d", what, e, m, mt_rank_w(ew, m), rank);
            int ht, hc;
            mt_route_pos_w(C.slot_tile0[first_slot[e]], ew, m, ht, hc);
            EXPECT(ht == C.tok_slot[m][k] && hc == C.tok_col[m][k], "%s: mt_route_pos_w(e %d, m %d) = (%d,%d), plan (%d,%d)", what, e, m, ht, hc, C.tok_slot[m][k], C.tok_col[m][k]);
        }
    if (max_slots) *max_slots = std::max(*max_slots, C.n_slots);
    if (max_tiles) *max_tiles = std::max(*max_tiles, C.n_tiles);
    if (max_chunks) *max_chunks = std::max(*max_chunks, chunks_max);
}

// The toy pipeline of T4 on three paths: the naive per-token MoE, the 8-row-tile path (every group of 8 tokens is its own M <= 8 call: one tile per
// expert), and the chunked path (all M rows in one call, one slot per 32-rank chunk). All three sum the same terms in the same order -> bit-equal.
struct Toy
{
    static constexpr int K = 32, I = 16, NO = 24;
    std::vector<double> x, W1, W3, W2;
    Toy(std::mt19937_64& g, int M, int NE) : x(static_cast<size_t>(M) * K), W1(static_cast<size_t>(NE) * I * K), W3(W1.size()), W2(static_cast<size_t>(NE) * NO * I)
    {
        std::uniform_real_distribution<double> u(-1, 1);
        for (auto& v : x) v = u(g);
        for (auto& v : W1) v = u(g);
        for (auto& v : W3) v = u(g);
        for (auto& v : W2) v = u(g);
    }
    void act(int e, int m, double* a) const
    {
        for (int i = 0; i < I; ++i)
        {
            double gt = 0, up = 0;
            for (int k = 0; k < K; ++k) { gt += W1[(static_cast<size_t>(e) * I + i) * K + k] * x[static_cast<size_t>(m) * K + k]; up += W3[(static_cast<size_t>(e) * I + i) * K + k] * x[static_cast<size_t>(m) * K + k]; }
            a[i] = gt / (1.0 + std::exp(-gt)) * up;
        }
    }
    double down(int e, const double* a, int o) const { double y = 0; for (int i = 0; i < I; ++i) y += W2[(static_cast<size_t>(e) * NO + o) * I + i] * a[i]; return y; }
    // run a plan over the tokens m0..m0+M-1 (the plan's token ids are relative to m0); returns out[m][o] for those rows
    template <int TK, int MM>
    void run(const PlanMT<TK, MM>& P, int m0, int M, std::vector<double>& out) const
    {
        std::vector<double> yp(static_cast<size_t>(P.n_tiles) * 8 * NO, std::nan(""));
        for (int s = 0; s < P.n_slots; ++s)
            for (int j = 0; j < P.slot_nt[s]; ++j)
            {
                const int t = P.slot_tile0[s] + j;
                for (int c = 0; c < 8; ++c)
                {
                    if (!((P.tile_mask[t] >> c) & 1)) continue;
                    double a[I];
                    act(P.slot_expert[s], m0 + P.tile_tok[t][c], a);
                    for (int o = 0; o < NO; ++o) yp[(static_cast<size_t>(t) * 8 + c) * NO + o] = P.tile_w[t][c] * down(P.slot_expert[s], a, o);
                }
            }
        for (int m = 0; m < M; ++m)
            for (int o = 0; o < NO; ++o)
            {
                double got = 0;
                for (int k = 0; k < TK; ++k) got += yp[(static_cast<size_t>(P.tok_slot[m][k]) * 8 + P.tok_col[m][k]) * NO + o];      // k_combine's formula
                out[static_cast<size_t>(m0 + m) * NO + o] = got;
            }
    }
};

template <int TK, int MM>
static void t6_pipeline(std::mt19937_64& g, int M, int NE, Kind kind)
{
    static int ex[MM][8]; static float w[MM][8];
    gen(kind, g, M, NE, TK, ex, w);
    Toy toy(g, M, NE);
    static PlanMT<TK, MM> P;
    std::vector<double> chunked(static_cast<size_t>(M) * Toy::NO), tile8(chunked.size()), naive(chunked.size());
    plan_ref_chunked<TK, MM>(M, ex, w, P);
    toy.run<TK, MM>(P, 0, M, chunked);
    for (int m0 = 0; m0 < M; m0 += 8)                                          // the 8-row-tile path: M <= 8 calls
    {
        const int gm = std::min(8, M - m0);
        plan_ref<TK, MM>(gm, ex + m0, w + m0, P);
        toy.run<TK, MM>(P, m0, gm, tile8);
    }
    for (int m = 0; m < M; ++m)
        for (int o = 0; o < Toy::NO; ++o)
        {
            double want = 0;
            for (int k = 0; k < TK; ++k) { double a[Toy::I]; toy.act(ex[m][k], m, a); want += w[m][k] * toy.down(ex[m][k], a, o); }
            naive[static_cast<size_t>(m) * Toy::NO + o] = want;
        }
    long bad8 = 0, badn = 0;
    for (size_t i = 0; i < chunked.size(); ++i) { bad8 += !(chunked[i] == tile8[i]); badn += !(chunked[i] == naive[i]); }
    EXPECT(bad8 == 0 && badn == 0, "pipeline W=%d %d/%d %s: %ld outputs differ from the 8-row-tile path, %ld from the naive MoE", M, NE, TK, kname[kind], bad8, badn);
}

// B-fragment gather of T3 over the chunked plan: the slot tiles are the same tiles, the table has ceil(M/32)*4 tiles (zero past M).
template <int TK, int MM>
static void t6_gather(std::mt19937_64& g, int K1, int M, int NE, Kind kind)
{
    static int ex[MM][8]; static float w[MM][8];
    gen(kind, g, M, NE, TK, ex, w);
    static PlanMT<TK, MM> P;
    plan_ref_chunked<TK, MM>(M, ex, w, P);
    const int ts = K1 * 4, NT = ((M + 31) / 32) * 4;                           // the adapter's xfrag size
    std::vector<uint32_t> table(static_cast<size_t>(NT) * ts, 0xdeadbeefu);
    for (int i = 0; i < NT * ts; ++i)
    {
        const MtFrag f = mt_frag_decode(i, K1);
        const int tok = 8 * f.tile + f.col;
        table[i] = tok < M ? static_cast<uint32_t>(tok) * (1u << 20) + static_cast<uint32_t>(f.k0) : 0u;
    }
    for (int s = 0; s < P.n_slots; ++s)
        for (int j = 0; j < P.slot_nt[s]; ++j)
        {
            const int t = P.slot_tile0[s] + j;
            for (int lane = 0; lane < 32; ++lane)
            {
                const int tau = P.tile_tok[t][lane >> 2];
                const int off = mt_xoff(tau, lane, ts);
                EXPECT(off >= 0 && off + (K1 / 16 - 1) * 64 + 1 < NT * ts, "W=%d gather offset out of the table", M);
                const int want_tok = tau < 0 ? 0 : tau;
                for (int ks = 0; ks < K1 / 16; ++ks)
                    for (int reg = 0; reg < 2; ++reg)
                    {
                        const uint32_t v = table[off + ks * 64 + reg];
                        const uint32_t k0 = static_cast<uint32_t>(ks * 16 + reg * 8 + 2 * (lane & 3));
                        EXPECT(v == static_cast<uint32_t>(want_tok) * (1u << 20) + k0, "W=%d gather tile %d lane %d ks %d reg %d wrong", M, t, lane, ks, reg);
                    }
            }
        }
}

template <int TK, int MM>
static void t6_generated(std::mt19937_64& g, int NE, std::initializer_list<int> ms, int reps, long& cases, int& mslots, int& mtiles, int& mchunks)
{
    for (int kind = 0; kind < KINDS; ++kind)
        for (int M : ms)
            for (int rep = 0; rep < reps; ++rep)
            {
                static int ex[MM][8]; static float w[MM][8];
                gen(static_cast<Kind>(kind), g, M, NE, TK, ex, w);
                char what[96]; snprintf(what, sizeof what, "%d/%d MM%d %s W=%d", NE, TK, MM, kname[kind], M);
                check_plan_chunked<TK, MM>(ex, w, M, NE, what, &mslots, &mtiles, &mchunks);
                ++cases;
            }
}

int main(int argc, char** argv)
{
    std::mt19937_64 g(20261025);
    t1_frag();

    // T2 + T5 over generated routes
    long cases = 0;
    int ms6 = 0, mt6 = 0, ms3 = 0, mt3 = 0;
    for (int kind = 0; kind < KINDS; ++kind)
        for (int M = 1; M <= 32; ++M)
            for (int rep = 0; rep < 12; ++rep)
            {
                static int ex[32][8]; static float w[32][8];
                gen(static_cast<Kind>(kind), g, M, 384, 6, ex, w);
                char what[96]; snprintf(what, sizeof what, "384/6 %s M=%d", kname[kind], M);
                check_plan<6, 32>(ex, w, M, 384, what, &ms6, &mt6);
                gen(static_cast<Kind>(kind), g, M, 128, 3, ex, w);
                snprintf(what, sizeof what, "128/3 %s M=%d", kname[kind], M);
                check_plan<3, 32>(ex, w, M, 128, what, &ms3, &mt3);
                cases += 2;
            }
    printf("T2 plan invariants + helper agreement: %ld cases (5 route kinds x M 1..32 x 12 seeds x 2 geometries) -> %s\n", cases, g_fail ? "FAIL" : "OK");
    EXPECT(ms6 == 192 && mt6 == 192, "T5 384/6: the all-distinct case should reach 192 slots / 192 tiles (got %d / %d)", ms6, mt6);
    EXPECT(ms3 == 96 && mt3 == 96, "T5 128/3: expected 96 / 96 (got %d / %d)", ms3, mt3);
    printf("T5 scratch bounds: max slots/tiles seen 384/6: %d/%d (bound 192), 128/3: %d/%d (bound 96)\n", ms6, mt6, ms3, mt3);

    // real routes (p128 reference dumps)
    const std::string dir = argc > 1 ? argv[1] : "";
    int real_cases = 0;
    for (int L : {0, 2, 20, 39})
    {
        FILE* f = dir.empty() ? nullptr : fopen((dir + "/L" + std::to_string(L) + ".router_idx.bin").c_str(), "rb");
        if (!f) continue;
        std::vector<int32_t> v(128 * 6);
        const size_t got = fread(v.data(), 4, v.size(), f);
        fclose(f);
        if (got != v.size()) continue;
        int max_nt = 0;
        for (int M = 1; M <= 32; ++M)
            for (int s = 0; s + M <= 128; s += 3)
            {
                static int ex[32][8]; static float w[32][8];
                for (int m = 0; m < M; ++m)
                    for (int k = 0; k < 8; ++k) { ex[m][k] = k < 6 ? v[(s + m) * 6 + k] : -1; w[m][k] = k < 6 ? 0.25f : 0.f; }
                char what[96]; snprintf(what, sizeof what, "p128 L%d M=%d @%d", L, M, s);
                check_plan<6, 32>(ex, w, M, 384, what);
                static PlanMT<6, 32> P; plan_ref<6, 32>(M, ex, w, P);
                for (int q = 0; q < P.n_slots; ++q) max_nt = std::max(max_nt, P.slot_nt[q]);
                ++real_cases;
            }
        printf("   real p128 L%d: max tiles per expert %d\n", L, max_nt);
    }
    printf("T2 real routes (p128): %d windows %s\n", real_cases, real_cases ? "checked" : "SKIPPED (no route dir)");

    // T3 gather, T4 pipeline
    for (int kind = 0; kind < KINDS; ++kind)
        for (int M : {1, 8, 9, 16, 17, 24, 31, 32})
        {
            t3_gather<6, 32>(g, M == 32 ? 5120 : 256, M, 384, static_cast<Kind>(kind));
            t3_gather<3, 32>(g, 128, M, 128, static_cast<Kind>(kind));
            for (int rep = 0; rep < 3; ++rep) { t4_pipeline<6, 32>(g, M, 384, static_cast<Kind>(kind)); t4_pipeline<3, 32>(g, M, 128, static_cast<Kind>(kind)); }
        }
    printf("T3 B-fragment gather (all slot tiles x 32 lanes x all k16 steps x 2 regs, 5 route kinds, M in {1,8,9,16,17,24,31,32}) -> %s\n", g_fail ? "see FAIL lines" : "OK");
    printf("T4 index chain slot->tile->column->act->ypart->combine, bit-equal to the naive per-token MoE (double) -> %s\n", g_fail ? "see FAIL lines" : "OK");
    // T6: prefill W = 33..256, chunked plan (k_plan_pw specification)
    {
        const int fail0 = g_fail;
        long cases = 0;
        int ms[4] = {0, 0, 0, 0}, mt[4] = {0, 0, 0, 0}, mc[4] = {0, 0, 0, 0};     // 384/6 MM128, 384/6 MM256, 128/3 MM128, 128/3 MM256
        t6_generated<6, 128>(g, 384, {33, 40, 63, 64, 65, 96, 127, 128}, 6, cases, ms[0], mt[0], mc[0]);
        t6_generated<6, 256>(g, 384, {129, 160, 191, 192, 255, 256}, 6, cases, ms[1], mt[1], mc[1]);
        t6_generated<3, 128>(g, 128, {33, 40, 63, 64, 65, 96, 127, 128}, 6, cases, ms[2], mt[2], mc[2]);
        t6_generated<3, 256>(g, 128, {129, 160, 191, 192, 255, 256}, 6, cases, ms[3], mt[3], mc[3]);
        printf("T6a chunked plan vs plan_ref (tile fields equal, slots = 32-rank chunk expansion, tile cover, W-word mt_rank_w / mt_route_pos_w, bounds): %ld generated cases -> %s\n", cases, g_fail == fail0 ? "OK" : "FAIL");
        printf("    max slots / tiles / chunks per expert seen (bounds from pw_slot_bound / pw_tile_bound at the worst W): 384/6 MM128 %d/%d/%d (396/432), 384/6 MM256 %d/%d/%d (420/528), 128/3 MM128 %d/%d/%d, 128/3 MM256 %d/%d/%d\n",
               ms[0], mt[0], mc[0], ms[1], mt[1], mc[1], ms[2], mt[2], mc[2], ms[3], mt[3], mc[3]);
        EXPECT(pw_slot_bound(768, 384) == 396 && pw_tile_bound(768, 384) == 432 && pw_slot_bound(1536, 384) == 420 && pw_tile_bound(1536, 384) == 528, "bound constants W128/W256 (396/432, 420/528)");

        // real p128 routes: whole prompt W = 128, windows 33..128; W = 256 = two layers' routes of the same prompt stacked (synthetic: there are no real 256-token routes)
        std::vector<std::vector<int32_t>> real;
        for (int L : {0, 2, 20, 39})
        {
            FILE* f = dir.empty() ? nullptr : fopen((dir + "/L" + std::to_string(L) + ".router_idx.bin").c_str(), "rb");
            std::vector<int32_t> v(128 * 6);
            if (f && fread(v.data(), 4, v.size(), f) == v.size()) real.push_back(v);
            if (f) fclose(f);
        }
        int rc128 = 0, rc256 = 0;
        for (size_t a = 0; a < real.size(); ++a)
        {
            for (int M : {33, 64, 96, 128})
                for (int s = 0; s + M <= 128; s += (M == 128 ? 1 : 5))
                {
                    static int ex[128][8]; static float w[128][8];
                    for (int m = 0; m < M; ++m) for (int k = 0; k < 8; ++k) { ex[m][k] = k < 6 ? real[a][(s + m) * 6 + k] : -1; w[m][k] = k < 6 ? 0.25f : 0.f; }
                    char what[96]; snprintf(what, sizeof what, "p128 layer#%zu W=%d @%d", a, M, s);
                    check_plan_chunked<6, 128>(ex, w, M, 384, what);
                    ++rc128;
                }
            for (size_t b = 0; b < real.size(); ++b)
                for (int M : {160, 256})
                {
                    static int ex[256][8]; static float w[256][8];
                    for (int m = 0; m < M; ++m)
                        for (int k = 0; k < 8; ++k)
                        {
                            const std::vector<int32_t>& src = m < 128 ? real[a] : real[b];
                            ex[m][k] = k < 6 ? src[(m & 127) * 6 + k] : -1; w[m][k] = k < 6 ? 0.25f : 0.f;
                        }
                    char what[96]; snprintf(what, sizeof what, "p128 layer#%zu+#%zu stacked W=%d (synthetic)", a, b, M);
                    check_plan_chunked<6, 256>(ex, w, M, 384, what);
                    ++rc256;
                }
        }
        printf("T6b real p128 routes: %d W<=128 windows (whole prompt W=128 included), %d stacked W=160/256 (synthetic) %s\n", rc128, rc256, real.empty() ? "SKIPPED (no route dir)" : "checked");

        // pipeline: chunked all-W-rows call == the 8-row-tile path == naive, bit for bit; B-fragment gather over the chunked tiles
        for (int kind = 0; kind < KINDS; ++kind)
        {
            for (int M : {33, 64, 97, 128})
            {
                t6_gather<6, 128>(g, 256, M, 384, static_cast<Kind>(kind));
                t6_gather<3, 128>(g, 128, M, 128, static_cast<Kind>(kind));
                for (int rep = 0; rep < 2; ++rep) { t6_pipeline<6, 128>(g, M, 384, static_cast<Kind>(kind)); t6_pipeline<3, 128>(g, M, 128, static_cast<Kind>(kind)); }
            }
            for (int M : {129, 192, 255, 256})
            {
                t6_gather<6, 256>(g, 256, M, 384, static_cast<Kind>(kind));
                t6_gather<3, 256>(g, 128, M, 128, static_cast<Kind>(kind));
                for (int rep = 0; rep < 2; ++rep) { t6_pipeline<6, 256>(g, M, 384, static_cast<Kind>(kind)); t6_pipeline<3, 256>(g, M, 128, static_cast<Kind>(kind)); }
            }
        }
        t6_gather<6, 256>(g, 5120, 256, 384, CLUSTER);                       // production K1 once at the largest W
        printf("T6c toy pipeline (W 33..256, 5 route kinds, 2 geometries): chunked call == 8-row-tile path == naive, bit-equal; T6d B-fragment gather over chunked tiles (xfrag = ceil(W/32)*4 tiles) -> %s\n", g_fail == fail0 ? "OK" : "see FAIL lines");
        EXPECT(mt_nchunks(1) == 1 && mt_nchunks(32) == 1 && mt_nchunks(33) == 2 && mt_nchunks(256) == 8 && pw_slot_bound(256 * 6, 384) <= 420, "mt_nchunks / bound spot values");
    }
    printf("%s (%d failed checks)\n", g_fail ? "HOST TEST FAIL" : "HOST TEST ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
