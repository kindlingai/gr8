// gr8 DeepSeek-V4.1-Flash multi-tile MoE plan: M = 9..32 tokens per step. Our own code.
//
// One SLOT = one distinct routed expert. Its n tokens (ascending token id) fill nt = ceil(n/8) TILES of 8 columns, column = rank % 8; the
// CTA of the slot reads the expert's weight records once and runs one MMA per live tile. Tiles are numbered globally (slot order, then
// tile order), so the routed scratch is indexed act[tile][8][I], ypart[tile][8][NOUT] and the combine's formula
// ypart[(tok_slot*8 + tok_col)*NOUT] is the M <= 8 one with tok_slot = the global tile.
//
// This header has no CUDA dependency: the index functions are __host__ __device__ and are what the kernels call, the CPU test
// (tests/ds_multitile_host_test.cpp) calls the same functions; plan_ref is the sequential specification the device plan must equal.
#pragma once
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define DSM_HD __host__ __device__ __forceinline__
#else
#define DSM_HD inline
#endif

namespace dsr
{
constexpr int MT_MAX = 32;                // tokens per step on the multi-tile path
constexpr int MT_NTMAX = MT_MAX / 8;      // tiles per expert (at most 4)

template <int TK, int MM = MT_MAX>
struct PlanMT
{
    static constexpr int SM = MM * TK;    // distinct experts <= SM, tiles <= SM (an expert's tiles <= its tokens)
    int n_slots;                          // distinct experts this step
    int n_tiles;                          // total tiles = sum of slot_nt
    int pad_[2];
    int slot_expert[SM];                  // ascending expert id; -1 beyond n_slots
    int slot_tile0[SM];                   // first global tile of the slot
    int slot_nt[SM];                      // tiles of the slot, 1..MM/8
    int tile_mask[SM];                    // bit c = column c of the tile is live (always the low n bits)
    float tile_w[SM][8];                  // routing weight (x 1.5) of the token in column c, 0 if unused
    int tile_tok[SM][8];                  // token in column c, -1 = unused
    int tok_slot[MM][8];                  // GLOBAL TILE of route k of token m; -1 for k >= TK or m >= M (the name k_combine reads)
    int tok_col[MM][8];                   // column in that tile
    float tok_w[MM][8];
    int tok_expert[MM][8];                // selected experts in selection order (best key first)
};

DSM_HD int mt_popc(unsigned v)
{
#ifdef __CUDA_ARCH__
    return __popc(v);
#else
    return __builtin_popcount(v);
#endif
}
DSM_HD int mt_ntiles(int n) { return (n + 7) >> 3; }                               // tiles of an expert with n tokens
DSM_HD int mt_tile_mask(int n, int j)                                             // live-column mask of tile j of an expert with n tokens
{
    int c = n - 8 * j;
    c = c > 8 ? 8 : c;
    return c > 0 ? static_cast<int>((1u << c) - 1u) : 0;
}
DSM_HD int mt_rank(unsigned em, int m) { return mt_popc(em & ((1u << m) - 1u)); }  // position of token m among the expert's tokens (em = token mask)
DSM_HD void mt_route_pos(int tile0, unsigned em, int m, int& tile, int& col)       // where route (token m -> this expert) lives
{
    const int r = mt_rank(em, m);
    tile = tile0 + (r >> 3);
    col = r & 7;
}

// ---- W-row prefill (M up to PW_MAX): the same plan for M > 32 (k_plan_pw, ds_router.cuh).
// The expert token mask is MW = MM / 32 words (word m >> 5, bit m & 31). The tile layout is unchanged: the rank-r token of an expert sits in
// global tile tile0_e + r / 8, column r % 8, tile0_e = tiles of the experts below. What changes is the slot: an expert's tokens are cut into
// CHUNKS of PW_CHUNK = 32 ranks (4 tiles), and each chunk is one slot, so the NT = 4 kernels (mx4::k_fc1_mt / k_fc2_mt) run it as they run
// an M = 32 expert: slot_expert repeats per chunk, slot_tile0 = tile0_e + 4 c, slot_nt = min(4, tiles left).
constexpr int PW_CHUNK = MT_MAX;          // tokens per slot of the chunked plan
constexpr int PW_MAX = 256;               // rows per call on the chunked path
DSM_HD int mt_nchunks(int n) { return (n + PW_CHUNK - 1) / PW_CHUNK; }            // slots of an expert with n tokens
DSM_HD int mt_rank_w(const unsigned* em, int m)                                   // position of token m among the expert's tokens (em = MW words)
{
    int r = 0;
    for (int i = 0; i < (m >> 5); ++i) r += mt_popc(em[i]);
    return r + mt_popc(em[m >> 5] & ((1u << (m & 31)) - 1u));
}
DSM_HD void mt_route_pos_w(int tile0, const unsigned* em, int m, int& tile, int& col)
{
    const int r = mt_rank_w(em, m);
    tile = tile0 + (r >> 3);
    col = r & 7;
}
// Scratch bounds for R = M * TK routes over at most NE experts: an expert with n >= 1 tokens takes ceil(n / 32) slots and ceil(n / 8) tiles, and
// ceil(n / q) <= (n + q - 1) / q, so slots <= (R + 31 d) / 32 and tiles <= (R + 7 d) / 8 with d = min(NE, R) experts (slots also <= R).
DSM_HD int pw_slot_bound(int R, int NE)
{
    const int d = R < NE ? R : NE, b = (R + 31 * d) / 32;
    return b < R ? b : R;
}
DSM_HD int pw_tile_bound(int R, int NE)
{
    const int d = R < NE ? R : NE;
    return (R + 7 * d) / 8;
}

// fc1 B operand. Tile tables are [tile][K1/16][32 lanes][2 regs] u32, tile t = tokens 8t..8t+7 (tile 0 = mx4::build_xfrag's table), tile stride ts =
// K1 * 4 words. Lane l holds column n = l >> 2 and k pair 2 (l & 3) (+8 for reg 1). A slot lane reads the table of the token in its column:
DSM_HD int mt_xoff(int tok, int lane, int ts)                                     // word offset of lane's fragment stream for token tok (-1 -> token 0)
{
    const int t = tok < 0 ? 0 : tok;
    return (t >> 3) * ts + (((t & 7) * 4 + (lane & 3)) * 2);
}
struct MtFrag { int tile, ks, lane, reg, col, k0; };                              // what table word i holds
DSM_HD MtFrag mt_frag_decode(int i, int K1)
{
    MtFrag f;
    const int per = K1 * 4;
    f.tile = i / per;
    const int r = i % per;
    f.ks = r >> 6;
    f.lane = (r >> 1) & 31;
    f.reg = r & 1;
    f.col = f.lane >> 2;
    f.k0 = f.ks * 16 + f.reg * 8 + 2 * (f.lane & 3);        // first k of the bf16 / fp16 pair; token = 8 tile + col
    return f;
}

// The weights of one token's kept routes, the kernels' float ops: picked[k] = the score of pick k (selection order, best biased key first). Sum of the first
// kc in selection order, w[k] = picked[k] / (sum + 1e-20f) * scale (scale = dsr::ROUTE_SCALE); k >= kc -> 0. kc == tk is the uncut weights.
inline void route_weights_ref(const float* picked, int tk, int kc, float scale, float* w)
{
    float sum = 0.f;
    for (int k = 0; k < kc; ++k) sum += picked[k];
    for (int k = 0; k < tk; ++k) w[k] = k < kc ? picked[k] / (sum + 1e-20f) * scale : 0.f;
}

// (plain host function templates: no device code; nvcc parses them in its device pass too, so no #ifdef around them)
// Sequential specification of the plan. Inputs: the routes (expert, weight) of tokens 0..M-1, route k = TK-wide, selection order. Fills every
// field of P the way k_plan_mt does (including the initial values of fields no route writes). Written for clarity, independent of the helpers
// above, so the CPU test can check them against it.
// kc (default TK) = routes kept per token, the top-k cut (GR8_DS_MOE_TOPK): routes k >= kc are ABSENT: in no expert's token list, tok_slot -1, tok_col 0,
// tok_expert -1, tok_w 0 (the caller passes weight[m][k] already renormalized over the first kc picks: route_weights_ref below).
template <int TK, int MM>
void plan_ref(int M, const int (*expert)[8], const float (*weight)[8], PlanMT<TK, MM>& P, int kc = TK)
{
    constexpr int SM = PlanMT<TK, MM>::SM;
    std::memset(&P, 0, sizeof(P));
    for (int s = 0; s < SM; ++s)
    {
        P.slot_expert[s] = -1;
        for (int c = 0; c < 8; ++c) P.tile_tok[s][c] = -1;
    }
    for (int m = 0; m < MM; ++m)
        for (int k = 0; k < 8; ++k) { P.tok_slot[m][k] = -1; P.tok_expert[m][k] = -1; }
    int maxe = 0;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < kc; ++k) maxe = expert[m][k] > maxe ? expert[m][k] : maxe;
    int slot = 0, tile = 0;
    for (int e = 0; e <= maxe; ++e)
    {
        int toks[MM], n = 0;
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < kc; ++k)
                if (expert[m][k] == e) toks[n++] = m;                       // ascending token id (a token holds an expert once)
        if (!n) continue;
        P.slot_expert[slot] = e;
        P.slot_tile0[slot] = tile;
        P.slot_nt[slot] = (n + 7) / 8;
        for (int r = 0; r < n; ++r)
        {
            const int t = tile + r / 8, c = r % 8, m = toks[r];
            int kk = 0;
            while (expert[m][kk] != e) ++kk;
            P.tile_mask[t] |= 1 << c;
            P.tile_tok[t][c] = m;
            P.tile_w[t][c] = weight[m][kk];
            P.tok_slot[m][kk] = t;
            P.tok_col[m][kk] = c;
        }
        tile += (n + 7) / 8;
        ++slot;
    }
    P.n_slots = slot;
    P.n_tiles = tile;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < kc; ++k) { P.tok_w[m][k] = weight[m][k]; P.tok_expert[m][k] = expert[m][k]; }
}

// Sequential specification of k_plan_pw (M up to MM = 256): plan_ref's tile layout, one slot per 32-token chunk of an expert (see mt_nchunks above).
// Independent of the helpers above (plain loops), so the CPU test can check them against it.
// kc as plan_ref (routes k >= kc are absent).
template <int TK, int MM>
void plan_ref_chunked(int M, const int (*expert)[8], const float (*weight)[8], PlanMT<TK, MM>& P, int kc = TK)
{
    constexpr int SM = PlanMT<TK, MM>::SM;
    std::memset(&P, 0, sizeof(P));
    for (int s = 0; s < SM; ++s)
    {
        P.slot_expert[s] = -1;
        for (int c = 0; c < 8; ++c) P.tile_tok[s][c] = -1;
    }
    for (int m = 0; m < MM; ++m)
        for (int k = 0; k < 8; ++k) { P.tok_slot[m][k] = -1; P.tok_expert[m][k] = -1; }
    int maxe = 0;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < kc; ++k) maxe = expert[m][k] > maxe ? expert[m][k] : maxe;
    int slot = 0, tile = 0;
    for (int e = 0; e <= maxe; ++e)
    {
        int toks[MM], n = 0;
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < kc; ++k)
                if (expert[m][k] == e) toks[n++] = m;
        if (!n) continue;
        for (int c = 0; 32 * c < n; ++c)                                    // one slot per chunk of 32 ranks = 4 tiles
        {
            const int left = (n - 32 * c + 7) / 8;                          // tiles still to place from this chunk's first rank
            P.slot_expert[slot] = e;
            P.slot_tile0[slot] = tile + 4 * c;
            P.slot_nt[slot] = left < 4 ? left : 4;
            ++slot;
        }
        for (int r = 0; r < n; ++r)
        {
            const int t = tile + r / 8, c = r % 8, m = toks[r];
            int kk = 0;
            while (expert[m][kk] != e) ++kk;
            P.tile_mask[t] |= 1 << c;
            P.tile_tok[t][c] = m;
            P.tile_w[t][c] = weight[m][kk];
            P.tok_slot[m][kk] = t;
            P.tok_col[m][kk] = c;
        }
        tile += (n + 7) / 8;
    }
    P.n_slots = slot;
    P.n_tiles = tile;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < kc; ++k) { P.tok_w[m][k] = weight[m][k]; P.tok_expert[m][k] = expert[m][k]; }
}
}   // namespace dsr
