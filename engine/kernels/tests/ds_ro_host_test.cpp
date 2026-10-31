// CPU gate T7 of the read-once prefill MoE (ds_ro.h, ds_prefill_ro.cuh). No GPU, no CUDA headers.
//
//   g++ -O2 -std=c++17 -Wall -o ds_ro_host_test ds_ro_host_test.cpp && ./ds_ro_host_test [p1200_dir]
//   p1200_dir: reference-model route dumps of one 1200-token prompt (L<n>.router_idx.bin int32 [1200][6], L<n>.router_w.bin
//   float [1200][6], layers 0/2/20/39). Without it the real-route cases are skipped.
//
// What it checks:
//   T7a  planner: a host transliteration of k_route_ro -> k_scan_ro -> k_place_ro (same warp-scan algebra, the header helpers mt_ntiles / mt_tile_mask /
//        mt_rank_w / mt_route_pos_w / ro_last_live) writes a PlanMT that was poisoned first. Its live prefixes (slots < n_slots, tiles < n_tiles incl. the dead
//        columns of each last tile, routes m < M, k < TK) equal plan_ref bit for bit; n_slots <= ro_slots_bound, n_tiles <= pw_tile_bound.
//   T7f  L2 lookahead (GR8_DS_RO_L2LA): ro_pf_target / ro_fc*_pf_rec(s) / ro_pf_chunk against the records each CTA owns (check_pf, called from T7b).
//   T7b  work model: every (CTA, warp, tile group) of the real index functions (ro_fc1_work / ro_fc1_rec / ro_fc2_work / ro_fc2_rec / ro_tile /
//        ro_group_nt) is enumerated: each weight record of a touched expert is owned by exactly one CTA and read once per tile group by that CTA; an
//        untouched expert's records are read 0 times; each (live tile, chunk) of fc1 and (live tile, output tile) of fc2 is produced exactly once; the
//        routes rebuilt from the loops equal the selected routes, ascending token inside an expert. ro_fc1_rec / ro_fc2_rec equal the byte offsets of
//        k_fc1_mt / k_fc2_mt (gate vs up, chunk, k record).
//   T7c  toy pipeline: ypart[tile][col][ot] = H(expert, token, ot) written through the fc2 loops, read through the combine's tok_slot / tok_col: equals
//        H(expert, token, ot) for every route.
//   T7d  modeled bytes (printed): DRAM-once and L2 -> SM weight bytes per layer and NTG (D, from the loops; formula cross-checked).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <set>
#include <map>
#include "../ds_plan_mt.h"
#include "../ds_ro.h"
using namespace dsr;

static int g_fail = 0;
#define EXPECT(c, ...) do { if (!(c)) { if (g_fail < 40) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } ++g_fail; } } while (0)

// production TP4 slice geometry
constexpr int K1 = 5120, IW = 576, NOUT = 5120, KG_ = 64, GR_ = 544;
constexpr int NCH = IW / 16, NG1 = K1 / KG_, OT = NOUT / 16, NG2 = IW / KG_;
constexpr double EXPERT_BYTES = double(NCH) * NG1 * 2 * GR_ + double(OT) * NG2 * GR_;       // 4,700,160

// ------------------------------------------------------------------------------------------------ routes
enum Kind { UNIFORM, CLUSTER, IDENTICAL, HOT, DISTINCT, KINDS };
static const char* kname[KINDS] = {"uniform", "cluster", "identical", "hot-expert", "all-distinct"};
static void gen(Kind kind, std::mt19937_64& g, int M, int NE, int TK, int (*ex)[8], float (*w)[8])
{
    std::vector<int> pool(NE);
    for (int i = 0; i < NE; ++i) pool[i] = i;
    std::shuffle(pool.begin(), pool.end(), g);
    const int P = kind == CLUSTER ? TK + 2 + static_cast<int>(g() % 6) : kind == IDENTICAL ? TK : NE;
    const int hot = pool[0];
    for (int m = 0; m < M; ++m)
    {
        std::vector<int> cand(pool.begin(), pool.begin() + P);
        if (kind == DISTINCT)
            for (int k = 0; k < TK; ++k) ex[m][k] = pool[(m * TK + k) % NE];
        else
        {
            if (kind == HOT) cand.erase(std::remove(cand.begin(), cand.end(), hot), cand.end());
            std::shuffle(cand.begin(), cand.end(), g);
            int k = 0;
            if (kind == HOT) ex[m][k++] = hot;
            for (; k < TK; ++k) ex[m][k] = cand[k - (kind == HOT ? 1 : 0)];
            if (kind == IDENTICAL) std::shuffle(ex[m], ex[m] + TK, g);
        }
        double s = 0;
        for (int k = 0; k < TK; ++k) { w[m][k] = 0.1f + static_cast<float>(g() % 1000) / 1000.f; s += w[m][k]; }
        for (int k = 0; k < TK; ++k) w[m][k] = static_cast<float>(w[m][k] / s * 1.5);
        for (int k = TK; k < 8; ++k) { ex[m][k] = -1; w[m][k] = 0.f; }
    }
}

// ------------------------------------------------------------------------------------------------ T7a: the planner kernels, on the host
// k_route_ro: tok_expert / tok_w, atomicOr of the token bit. k_scan_ro: per-expert popcount, two 32-lane Hillis-Steele scans, block combine through
// wcount / wtiles. k_place_ro: rank by popcount of the lower words. The plan is poisoned first (0xA5 bytes): the kernels may only write what they promise.
template <int NE, int TK, int MM>
static void emu_plan(int M, const int (*ex)[8], const float (*w)[8], PlanMT<TK, MM>& P, std::vector<unsigned>& emask, std::vector<int>& tile0_of)
{
    constexpr int MW = MM / 32, RW = NE / 32;
    std::memset(&P, 0xA5, sizeof(P));
    emask.assign(static_cast<size_t>(NE) * MW, 0u);                                  // cudaMemsetAsync
    tile0_of.assign(NE, 0x5a5a5a5a);
    for (int m = 0; m < M; ++m)                                                       // k_route_ro
        for (int l = 0; l < TK; ++l)
        {
            P.tok_expert[m][l] = ex[m][l];
            P.tok_w[m][l] = w[m][l];
            emask[static_cast<size_t>(ex[m][l]) * MW + (m >> 5)] |= 1u << (m & 31);
        }
    std::vector<int> n(NE), nt(NE), in_s(NE), in_t(NE), wcount(RW), wtiles(RW);       // k_scan_ro
    for (int t = 0; t < NE; ++t)
    {
        n[t] = 0;
        for (int i = 0; i < MW; ++i) n[t] += mt_popc(emask[static_cast<size_t>(t) * MW + i]);
        nt[t] = mt_ntiles(n[t]);
        in_s[t] = n[t] ? 1 : 0;
        in_t[t] = nt[t];
    }
    for (int wv = 0; wv < RW; ++wv)
        for (int o = 1; o < 32; o <<= 1)
        {
            int vs[32], vt[32];
            for (int l = 0; l < 32; ++l) { vs[l] = l >= o ? in_s[wv * 32 + l - o] : 0; vt[l] = l >= o ? in_t[wv * 32 + l - o] : 0; }
            for (int l = o; l < 32; ++l) { in_s[wv * 32 + l] += vs[l]; in_t[wv * 32 + l] += vt[l]; }
        }
    for (int wv = 0; wv < RW; ++wv) { wcount[wv] = in_s[wv * 32 + 31]; wtiles[wv] = in_t[wv * 32 + 31]; }
    for (int t = 0; t < NE; ++t)
    {
        int sb = 0, tb = 0;
        for (int i = 0; i < t / 32; ++i) { sb += wcount[i]; tb += wtiles[i]; }
        int tile0 = 0;
        if (n[t])
        {
            const int slot = sb + in_s[t] - 1;
            tile0 = tb + in_t[t] - nt[t];
            P.slot_expert[slot] = t;
            P.slot_tile0[slot] = tile0;
            P.slot_nt[slot] = nt[t];
            for (int j = 0; j < nt[t]; ++j) P.tile_mask[tile0 + j] = mt_tile_mask(n[t], j);
            const int last = tile0 + nt[t] - 1;
            for (int c = ro_last_live(n[t]); c < 8; ++c) { P.tile_tok[last][c] = -1; P.tile_w[last][c] = 0.f; }
        }
        tile0_of[t] = tile0;
        if (t == NE - 1) { P.n_slots = sb + in_s[t]; P.n_tiles = tb + in_t[t]; }
    }
    for (int r = 0; r < M * TK; ++r)                                                  // k_place_ro
    {
        const int m = r / TK, k = r % TK, e = P.tok_expert[m][k];
        int tile, col;
        mt_route_pos_w(tile0_of[e], &emask[static_cast<size_t>(e) * MW], m, tile, col);
        P.tok_slot[m][k] = tile;
        P.tok_col[m][k] = col;
        P.tile_w[tile][col] = P.tok_w[m][k];
        P.tile_tok[tile][col] = m;
    }
}

// live-prefix comparison against plan_ref (the unchunked specification)
template <int TK, int MM>
static void compare_plan(int M, int NE, const PlanMT<TK, MM>& R, const PlanMT<TK, MM>& E, const char* what)
{
    EXPECT(R.n_slots == E.n_slots && R.n_tiles == E.n_tiles, "%s: n_slots/n_tiles %d/%d, spec %d/%d", what, E.n_slots, E.n_tiles, R.n_slots, R.n_tiles);
    if (R.n_slots != E.n_slots || R.n_tiles != E.n_tiles || R.n_slots < 0 || R.n_slots > PlanMT<TK, MM>::SM) return;
    EXPECT(E.n_slots <= ro_slots_bound(M * TK, NE) && E.n_tiles <= pw_tile_bound(M * TK, NE), "%s: bounds slots %d/%d tiles %d/%d", what, E.n_slots, ro_slots_bound(M * TK, NE), E.n_tiles, pw_tile_bound(M * TK, NE));
    for (int s = 0; s < R.n_slots; ++s)
        EXPECT(R.slot_expert[s] == E.slot_expert[s] && R.slot_tile0[s] == E.slot_tile0[s] && R.slot_nt[s] == E.slot_nt[s], "%s: slot %d (%d,%d,%d) spec (%d,%d,%d)", what, s,
               E.slot_expert[s], E.slot_tile0[s], E.slot_nt[s], R.slot_expert[s], R.slot_tile0[s], R.slot_nt[s]);
    for (int t = 0; t < R.n_tiles; ++t)
    {
        EXPECT(R.tile_mask[t] == E.tile_mask[t], "%s: tile %d mask %x spec %x", what, t, E.tile_mask[t], R.tile_mask[t]);
        for (int c = 0; c < 8; ++c)
            EXPECT(R.tile_tok[t][c] == E.tile_tok[t][c] && std::memcmp(&R.tile_w[t][c], &E.tile_w[t][c], 4) == 0, "%s: tile %d col %d tok %d w %g, spec tok %d w %g", what, t, c, E.tile_tok[t][c], E.tile_w[t][c], R.tile_tok[t][c], R.tile_w[t][c]);
    }
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < TK; ++k)
            EXPECT(R.tok_slot[m][k] == E.tok_slot[m][k] && R.tok_col[m][k] == E.tok_col[m][k] && R.tok_expert[m][k] == E.tok_expert[m][k] &&
                   std::memcmp(&R.tok_w[m][k], &E.tok_w[m][k], 4) == 0, "%s: route (%d,%d) mismatch", what, m, k);
}

// ------------------------------------------------------------------------------------------------ T7b / T7c: the expert loops
struct Cfg1 { int KS, CW, NTG; };
struct Cfg2 { int WARPS, NTG; };
struct Model { double dram = 0, l2sm = 0, l2sm_formula = 0; int touched = 0, groups_max = 0; };

static inline uint64_t H(int e, int tok, int ot) { return (uint64_t(e) * 1000003ull + uint64_t(tok)) * 131ull + uint64_t(ot) + 7ull; }

// T7f: the GR8_DS_RO_L2LA ranges. The kernels call ro_pf_target / ro_fc*_pf_rec(s) / ro_pf_chunk (ds_ro.h); here the same functions are checked against the records the loops above
// really assigned to each CTA (lo / hi / cnt per block = the distinct records it owns): CTA b + la prefetches exactly the contiguous byte range CTA b + la reads.
// startf(expert, cgrp) / recs = the kernel's range (first record, record count); order / slot_expert = the launch map the kernel indexes.
template <class StartF>
static void check_pf(const char* what, const char* kn, int ups, int ns, int sb, const int* order, const int* slot_expert, const std::vector<size_t>& lo, const std::vector<size_t>& hi, const std::vector<size_t>& cnt, StartF startf, size_t recs)
{
    const int nblk = ns * ups;                                                         // launched blocks; the rest return before the prefetch
    for (int la : {-3, 0, 1, 7, 48, 96, 100000})
    {
        long valid = 0, bad_t = 0, bad_range = 0, bad_bytes = 0;
        for (int b = 0; b < sb * ups && b < nblk; ++b)
        {
            int bp = -1;
            const bool v = ro_pf_target(b, la, ups, ns, bp);
            const bool want_v = la > 0 && (b + la) / ups < ns;                          // independent: the position b + la falls in a launched slot group
            bad_t += (v != want_v) || (v && bp != b + la);
            if (!v) continue;
            ++valid;
            if (bp < 0 || bp >= nblk) { ++bad_range; continue; }                           // a target beyond the launched blocks: fail here, do not index past the per-block tables
            const int slot = ro_pos_slot(order, bp / ups), e = slot_expert[slot];
            const size_t start = startf(e, bp % ups);
            bad_range += start != lo[bp] || recs != cnt[bp] || hi[bp] - lo[bp] + 1 != cnt[bp];
            const unsigned bytes = static_cast<unsigned>(recs * GR_);
            unsigned expect_off = 0, off = 0, sz = 0;
            for (int c = 0; c < 4096 && ro_pf_chunk(bytes, c, off, sz); ++c)             // tiles [0, bytes) chunk by chunk (more chunks than lanes is only legal outside the production geometry: the kernels static_assert it)
            {
                bad_bytes += off != expect_off || sz == 0 || sz > RO_PF_CHUNK || off % 16 || sz % 16;
                expect_off += sz;
            }
            bad_bytes += expect_off != bytes || (start * GR_) % 16;
        }
        const long want_valid = la > 0 && nblk > la ? nblk - la : 0;
        EXPECT(bad_t == 0 && valid == want_valid, "%s: %s L2LA la=%d: target flag wrong at %ld blocks, %ld valid targets != %ld", what, kn, la, bad_t, valid, want_valid);
        EXPECT(bad_range == 0, "%s: %s L2LA la=%d: %ld prefetch ranges differ from the records the CTA ahead reads", what, kn, la, bad_range);
        EXPECT(bad_bytes == 0, "%s: %s L2LA la=%d: %ld byte chunks not contiguous / 16-byte aligned / within 8 KiB", what, kn, la, bad_bytes);
    }
}

template <int TK, int MM>
static void check_work(const PlanMT<TK, MM>& P, int M, int NE, const int (*ex)[8], const Cfg1& c1, const Cfg2& c2, const char* what, Model* model, bool toy, const int* order = nullptr)
{
    const int sb = ro_slots_bound(M * TK, NE), ns = P.n_slots, nti = P.n_tiles;
    EXPECT(ns <= sb, "%s: n_slots %d beyond the launch bound %d", what, ns, sb);
    for (int pos = 0; pos < ns; ++pos)                                                // the launch map: position -> slot is the order array, or the identity without one
        EXPECT(ro_pos_slot(order, pos) == (order ? order[pos] : pos), "%s: ro_pos_slot(order, %d) = %d, the launch map says %d", what, pos, ro_pos_slot(order, pos), order ? order[pos] : pos);
    std::vector<char> touched(NE, 0);
    for (int s = 0; s < ns; ++s) touched[P.slot_expert[s]] = 1;
    std::vector<int> ngr(NE, 0);
    for (int s = 0; s < ns; ++s) ngr[P.slot_expert[s]] = ro_ngroups(P.slot_nt[s], c1.NTG);
    // routes the loops must reproduce
    std::set<std::pair<int, int>> want;
    for (int m = 0; m < M; ++m) for (int k = 0; k < TK; ++k) want.insert({ex[m][k], m});
    const size_t nrec1 = size_t(NE) * NCH * NG1 * 2, nrec2 = size_t(NE) * OT * NG2;

    // ---- fc1
    {
        const int UPS = NCH / c1.CW, NW = c1.KS * c1.CW, NGP = NG1 / c1.KS;
        std::vector<int> owner(nrec1, -1);
        std::vector<uint16_t> reads(nrec1, 0);
        std::vector<uint8_t> out(size_t(nti + 2) * NCH, 0);
        std::vector<std::pair<int, int>> got;
        std::vector<int> launch_e;
        std::vector<size_t> plo(size_t(sb) * UPS, SIZE_MAX), phi(size_t(sb) * UPS, 0), pcnt(size_t(sb) * UPS, 0);       // records each block OWNS (first assignment), for T7f
        long dup = 0;
        double l2 = 0;
        for (int block = 0; block < sb * UPS; ++block)
        {
            const int pos = block / UPS;
            if (pos >= ns) continue;                                                   // early exit on *n_slots_dev (the position, before the order is read)
            const int slot = ro_pos_slot(order, pos);                                  // GR8_DS_RO_ORDER: the CTA group at `pos` works slot order[pos]
            if (block % UPS == 0) launch_e.push_back(P.slot_expert[slot]);              // the expert whose weights the CTA group launched at position `pos` streams
            const int e = P.slot_expert[slot], nt = P.slot_nt[slot], tile0 = P.slot_tile0[slot], ng = ro_ngroups(nt, c1.NTG);
            for (int warp = 0; warp < NW; ++warp)
            {
                int slot_w, chunk, part;
                ro_fc1_work(block, warp, UPS, c1.KS, c1.CW, slot_w, chunk, part);
                EXPECT(slot_w == pos && chunk >= 0 && chunk < NCH && part >= 0 && part < c1.KS, "%s: fc1 work map (block %d warp %d) -> slot %d chunk %d part %d", what, block, warp, slot_w, chunk, part);
                if (slot_w != pos || chunk < 0 || chunk >= NCH || part < 0 || part >= c1.KS) return;
                // the kernel's weight pointer: base rec of (e, chunk, part * NGP) then +2 per record, +1 for the up half (fc1_part_mt: p * 2 * GR, + GR)
                const size_t base = ro_fc1_rec(e, chunk, part * NGP, 0, NCH, NG1);
                for (int g = 0; g < ng; ++g)
                {
                    const int ntg = ro_group_nt(nt, c1.NTG, g), t0 = ro_tile(tile0, g, c1.NTG, 0);
                    for (int p = 0; p < NGP; ++p)
                        for (int up = 0; up < 2; ++up)
                        {
                            const size_t id = base + size_t(p) * 2 + up;
                            if (id >= nrec1) { EXPECT(false, "%s: fc1 record %zu out of range", what, id); return; }
                            if (owner[id] < 0) { owner[id] = block; plo[block] = std::min<size_t>(plo[block], id); phi[block] = std::max<size_t>(phi[block], id); ++pcnt[block]; } else if (owner[id] != block) ++dup;
                            ++reads[id];
                            l2 += GR_;
                        }
                    if (part == 0)
                        for (int t = 0; t < ntg; ++t)
                        {
                            const int tile = ro_tile(tile0, g, c1.NTG, t);
                            if (tile < 0 || tile >= nti + 2) { EXPECT(false, "%s: fc1 tile %d out of range", what, tile); return; }
                            ++out[size_t(tile) * NCH + chunk];
                            if (chunk == 0)                                           // routes rebuilt from the loop (once per tile)
                                for (int c = 0; c < 8; ++c)
                                    if (tile < nti && (P.tile_mask[tile] >> c & 1)) got.push_back({e, P.tile_tok[tile][c]});
                        }
                    (void)t0;
                }
            }
        }
        EXPECT(dup == 0, "%s: fc1 %ld records read by more than one CTA", what, dup);
        check_pf(what, "fc1", UPS, ns, sb, order, P.slot_expert, plo, phi, pcnt, [&](int e, int cg) { return ro_fc1_pf_rec(e, cg, c1.CW, NCH, NG1); }, ro_fc1_pf_recs(c1.CW, NG1));
        {   // launch sequence: the CTA group launched at position p streams the expert of slot order[p] (identity without an order)
            long badl = (long)launch_e.size() != ns;
            for (int p = 0; p < ns && p < (int)launch_e.size(); ++p) badl += launch_e[p] != P.slot_expert[order ? order[p] : p];
            EXPECT(badl == 0, "%s: fc1 launch sequence differs from the launch map at %ld positions", what, badl);
        }
        long bad_owner = 0, bad_reads = 0, bad_untouched = 0;
        for (int e = 0; e < NE; ++e)
        {
            const size_t b0 = size_t(e) * NCH * NG1 * 2;
            for (size_t i = 0; i < size_t(NCH) * NG1 * 2; ++i)
            {
                const size_t id = b0 + i;
                if (touched[e]) { bad_owner += owner[id] < 0; bad_reads += reads[id] != ngr[e]; }
                else bad_untouched += reads[id] != 0;
            }
        }
        EXPECT(bad_owner == 0 && bad_reads == 0 && bad_untouched == 0, "%s: fc1 records: %ld of a touched expert unowned, %ld not read exactly ngroups times, %ld read in an untouched expert", what, bad_owner, bad_reads, bad_untouched);
        long bad_out = 0;
        for (int t = 0; t < nti + 2; ++t)
            for (int ch = 0; ch < NCH; ++ch) bad_out += out[size_t(t) * NCH + ch] != (t < nti ? 1 : 0);
        EXPECT(bad_out == 0, "%s: fc1 (tile, chunk) outputs: %ld not produced exactly once (or produced beyond n_tiles)", what, bad_out);
        // routes: same multiset as the selection; ascending token inside each expert
        std::set<std::pair<int, int>> gs(got.begin(), got.end());
        EXPECT(got.size() == want.size() && gs == want, "%s: routes rebuilt from the fc1 loop (%zu) != selected routes (%zu)", what, got.size(), want.size());
        std::map<int, int> last;
        for (auto& pr : got) { auto it = last.find(pr.first); EXPECT(it == last.end() || it->second < pr.second, "%s: expert %d tokens not ascending", what, pr.first); last[pr.first] = pr.second; }
        if (model)
        {
            model->l2sm += l2;
            double f = 0;
            for (int e = 0; e < NE; ++e) f += touched[e] ? double(NCH) * NG1 * 2 * GR_ * ngr[e] : 0;
            model->l2sm_formula += f;
            EXPECT(std::fabs(f - l2) < 1.0, "%s: fc1 L2->SM bytes from the loops %.0f != formula %.0f", what, l2, f);
        }
    }
    // ---- fc2 (+ toy ypart)
    {
        const int UPS = OT / c2.WARPS;
        std::vector<int> owner(nrec2, -1);
        std::vector<uint16_t> reads(nrec2, 0);
        std::vector<uint8_t> out(size_t(nti + 2) * OT, 0);
        std::vector<uint64_t> ypart;
        if (toy) ypart.assign(size_t(nti) * 8 * OT, ~0ull);
        std::vector<int> launch_e;
        std::vector<size_t> plo(size_t(sb) * UPS, SIZE_MAX), phi(size_t(sb) * UPS, 0), pcnt(size_t(sb) * UPS, 0);
        long dup = 0;
        double l2 = 0;
        for (int block = 0; block < sb * UPS; ++block)
        {
            const int pos = block / UPS;
            if (pos >= ns) continue;
            const int slot = ro_pos_slot(order, pos);
            if (block % UPS == 0) launch_e.push_back(P.slot_expert[slot]);
            const int e = P.slot_expert[slot], nt = P.slot_nt[slot], tile0 = P.slot_tile0[slot], ng = ro_ngroups(nt, c2.NTG);
            for (int warp = 0; warp < c2.WARPS; ++warp)
            {
                int slot_w, ot;
                ro_fc2_work(block, warp, UPS, c2.WARPS, slot_w, ot);
                EXPECT(slot_w == pos && ot >= 0 && ot < OT, "%s: fc2 work map (block %d warp %d) -> slot %d ot %d", what, block, warp, slot_w, ot);
                if (slot_w != pos || ot < 0 || ot >= OT) return;
                for (int g = 0; g < ng; ++g)
                {
                    const int ntg = ro_group_nt(nt, c2.NTG, g);
                    for (int rec = 0; rec < NG2; ++rec)
                    {
                        const size_t id = ro_fc2_rec(e, ot, rec, OT, NG2);
                        if (id >= nrec2) { EXPECT(false, "%s: fc2 record %zu out of range", what, id); return; }
                        if (owner[id] < 0) { owner[id] = block; plo[block] = std::min<size_t>(plo[block], id); phi[block] = std::max<size_t>(phi[block], id); ++pcnt[block]; } else if (owner[id] != block) ++dup;
                        ++reads[id];
                        l2 += GR_;
                    }
                    for (int t = 0; t < ntg; ++t)
                    {
                        const int tile = ro_tile(tile0, g, c2.NTG, t);
                        if (tile < 0 || tile >= nti + 2) { EXPECT(false, "%s: fc2 tile %d out of range", what, tile); return; }
                        ++out[size_t(tile) * OT + ot];
                        if (toy && tile < nti)
                            for (int c = 0; c < 8; ++c)
                                if (P.tile_mask[tile] >> c & 1) ypart[(size_t(tile) * 8 + c) * OT + ot] = H(e, P.tile_tok[tile][c], ot);
                    }
                }
            }
        }
        EXPECT(dup == 0, "%s: fc2 %ld records read by more than one CTA", what, dup);
        check_pf(what, "fc2", UPS, ns, sb, order, P.slot_expert, plo, phi, pcnt, [&](int e, int og) { return ro_fc2_pf_rec(e, og, c2.WARPS, OT, NG2); }, ro_fc2_pf_recs(c2.WARPS, NG2));
        {   // launch sequence: the CTA group launched at position p streams the expert of slot order[p] (identity without an order)
            long badl = (long)launch_e.size() != ns;
            for (int p = 0; p < ns && p < (int)launch_e.size(); ++p) badl += launch_e[p] != P.slot_expert[order ? order[p] : p];
            EXPECT(badl == 0, "%s: fc2 launch sequence differs from the launch map at %ld positions", what, badl);
        }
        long bad_owner = 0, bad_reads = 0, bad_untouched = 0;
        std::vector<int> ng2(NE, 0);
        for (int s = 0; s < ns; ++s) ng2[P.slot_expert[s]] = ro_ngroups(P.slot_nt[s], c2.NTG);
        for (int e = 0; e < NE; ++e)
        {
            const size_t b0 = size_t(e) * OT * NG2;
            for (size_t i = 0; i < size_t(OT) * NG2; ++i)
            {
                const size_t id = b0 + i;
                if (touched[e]) { bad_owner += owner[id] < 0; bad_reads += reads[id] != ng2[e]; }
                else bad_untouched += reads[id] != 0;
            }
        }
        EXPECT(bad_owner == 0 && bad_reads == 0 && bad_untouched == 0, "%s: fc2 records: %ld of a touched expert unowned, %ld not read exactly ngroups times, %ld read in an untouched expert", what, bad_owner, bad_reads, bad_untouched);
        long bad_out = 0;
        for (int t = 0; t < nti + 2; ++t)
            for (int o = 0; o < OT; ++o) bad_out += out[size_t(t) * OT + o] != (t < nti ? 1 : 0);
        EXPECT(bad_out == 0, "%s: fc2 (tile, output tile) outputs: %ld not produced exactly once (or beyond n_tiles)", what, bad_out);
        if (toy)                                                                     // combine: ypart[(tok_slot * 8 + tok_col) * OT + ot] == H(expert, token, ot)
        {
            long bad = 0;
            for (int m = 0; m < M; ++m)
                for (int k = 0; k < TK; ++k)
                {
                    const int sl = P.tok_slot[m][k], co = P.tok_col[m][k];
                    if (sl < 0 || sl >= nti || co < 0 || co > 7) { ++bad; continue; }
                    for (int o = 0; o < OT; ++o) bad += ypart[(size_t(sl) * 8 + co) * OT + o] != H(ex[m][k], m, o);
                }
            EXPECT(bad == 0, "%s: toy combine: %ld ypart words differ from H(expert, token, ot)", what, bad);
        }
        if (model)
        {
            model->l2sm += l2;
            double f = 0;
            for (int e = 0; e < NE; ++e) f += touched[e] ? double(OT) * NG2 * GR_ * ng2[e] : 0;
            model->l2sm_formula += f;
            EXPECT(std::fabs(f - l2) < 1.0, "%s: fc2 L2->SM bytes from the loops %.0f != formula %.0f", what, l2, f);
        }
    }
    if (model)
    {
        int t = 0, gm = 0;
        for (int e = 0; e < NE; ++e) { t += touched[e]; gm = std::max(gm, ngr[e]); }
        model->touched = t;
        model->dram = double(t) * EXPERT_BYTES;
        model->groups_max = gm;
    }
}

// ------------------------------------------------------------------------------------------------ T7e: the launch order (GR8_DS_RO_ORDER, k_order_ro)
// emu_order = the kernel's algebra with the header functions: thread t computes ro_ord_key(t), then its position by counting the slots that run before it.
static std::vector<int> emu_order(const std::vector<int>& nt)
{
    const int ns = static_cast<int>(nt.size());
    std::vector<int> num(ns), den(ns), cls(ns), order(ns, -1);
    for (int t = 0; t < ns; ++t) ro_ord_key(nt.data(), ns, t, num[t], den[t], cls[t]);
    for (int t = 0; t < ns; ++t)
    {
        int pos = 0;
        for (int j = 0; j < ns; ++j) pos += (j != t) && ro_ord_before(num[j], den[j], cls[j], j, num[t], den[t], cls[t], t);
        EXPECT(pos >= 0 && pos < ns && order[pos] < 0, "order: position %d of slot %d taken or out of range", pos, t);
        if (pos >= 0 && pos < ns) order[pos] = t;
    }
    return order;
}
// independent reference: two explicit lists, prefix sums, one stable sort on (key, class)
static std::vector<int> ref_order(const std::vector<int>& nt)
{
    struct It { long long num, den; int cls, s; };
    std::vector<int> hv, lt;
    for (int s = 0; s < static_cast<int>(nt.size()); ++s) (nt[s] >= 5 ? hv : lt).push_back(s);      // 5 = RO_ORD_HEAVY_NT, written out
    std::stable_sort(hv.begin(), hv.end(), [&](int a, int b) { return nt[a] > nt[b]; });
    long long ht = 0, cum = 0;
    for (int s : hv) ht += nt[s];
    std::vector<It> it;
    for (int s : hv) { it.push_back({cum, ht, 0, s}); cum += nt[s]; }
    for (size_t i = 0; i < lt.size(); ++i) it.push_back({static_cast<long long>(i), static_cast<long long>(lt.size()), 1, lt[i]});
    std::stable_sort(it.begin(), it.end(), [](const It& a, const It& b) { const long long x = a.num * b.den, y = b.num * a.den; return x != y ? x < y : a.cls < b.cls; });
    std::vector<int> o;
    for (auto& i : it) o.push_back(i.s);
    return o;
}
// returns the order (empty if the plan has no slot) after checking it: a permutation, equal to the reference, heaviest expert first, the heavy ones by tiles descending
template <int TK, int MM>
static std::vector<int> check_order(const PlanMT<TK, MM>& P, const char* what, double* late = nullptr)
{
    const int ns = P.n_slots;
    std::vector<int> nt(P.slot_nt, P.slot_nt + ns);
    if (ns == 0) return {};
    const std::vector<int> o = emu_order(nt), r = ref_order(nt);
    std::vector<char> seen(ns, 0);
    for (int p = 0; p < ns; ++p) { EXPECT(o[p] >= 0 && o[p] < ns && !seen[o[p]], "%s: order is not a permutation at position %d", what, p); if (o[p] >= 0 && o[p] < ns) seen[o[p]] = 1; }
    EXPECT(o == r, "%s: order differs from the independent reference", what);
    int mx = 0;
    for (int s = 0; s < ns; ++s) mx = std::max(mx, nt[s]);
    if (mx >= 5) EXPECT(nt[o[0]] == mx, "%s: first slot has %d tiles, the heaviest has %d", what, nt[o[0]], mx);
    int lastheavy = -1, hv = 0;
    int prev_h = 1 << 30, prev_l = -1;
    for (int p = 0; p < ns; ++p)
    {
        const int s = o[p];
        if (nt[s] >= 5) { EXPECT(nt[s] <= prev_h, "%s: heavy slots not by tiles descending at %d", what, p); prev_h = nt[s]; lastheavy = p; ++hv; }
        else { EXPECT(s > prev_l, "%s: light slots not in slot order at %d", what, p); prev_l = s; }
    }
    if (late && hv) *late = static_cast<double>(lastheavy) / ns;
    return o;
}

// one route set: plan (emulated vs plan_ref), then the work model for each (fc1, fc2) config pair
template <int NE, int TK, int MM>
static void run_case(int M, const int (*ex)[8], const float (*w)[8], const char* what, const std::vector<std::pair<Cfg1, Cfg2>>& cfgs, bool toy, long& cases, Model* model = nullptr, int model_cfg = -1)
{
    static PlanMT<TK, MM> R, E;
    static std::vector<unsigned> emask;
    static std::vector<int> t0;
    plan_ref<TK, MM>(M, ex, w, R);
    emu_plan<NE, TK, MM>(M, ex, w, E, emask, t0);
    const int f0 = g_fail;
    compare_plan<TK, MM>(M, NE, R, E, what);
    {                                                                                 // T8: the distinct-expert counter's counts == the plan's n_slots / heaviest slot
        std::vector<int> cnt(NE);
        int d = 0, mx = 0, mxr = 0;
        elog_stats(ex, M, TK, NE, cnt.data(), d, mx);
        for (int s = 0; s < R.n_slots; ++s)
        {
            int n = 0;
            for (int j = 0; j < R.slot_nt[s]; ++j) n += mt_popc(static_cast<unsigned>(R.tile_mask[R.slot_tile0[s] + j]));
            mxr = n > mxr ? n : mxr;
        }
        EXPECT(d == R.n_slots && mx == mxr, "%s: counter distinct %d maxn %d, plan_ref n_slots %d heaviest %d", what, d, mx, R.n_slots, mxr);
        EXPECT(sizeof(ExpertLogRec) == 32 && offsetof(ExpertLog, rec) == 32, "ExpertLog layout");
    }
    ++cases;
    if (g_fail != f0) return;                                                         // the work model reads the (broken) plan: stop at the first layer of failure
    for (size_t i = 0; i < cfgs.size(); ++i)
    {
        char tag[200]; snprintf(tag, sizeof tag, "%s fc1(KS%d,CW%d,NTG%d) fc2(W%d,NTG%d)", what, cfgs[i].first.KS, cfgs[i].first.CW, cfgs[i].first.NTG, cfgs[i].second.WARPS, cfgs[i].second.NTG);
        check_work<TK, MM>(E, M, NE, ex, cfgs[i].first, cfgs[i].second, tag, static_cast<int>(i) == model_cfg ? model : nullptr, toy && i == 0);
        ++cases;
    }
    if (model) return;                                                                // the byte model (T7d) is order independent
    const std::vector<int> ord = check_order<TK, MM>(E, what);                         // T7e: the order, then the whole work model again through it
    ++cases;
    if (ord.empty()) return;
    for (size_t i = 0; i < cfgs.size(); ++i)
    {
        char tag[220]; snprintf(tag, sizeof tag, "%s +order fc1(KS%d,CW%d,NTG%d) fc2(W%d,NTG%d)", what, cfgs[i].first.KS, cfgs[i].first.CW, cfgs[i].first.NTG, cfgs[i].second.WARPS, cfgs[i].second.NTG);
        check_work<TK, MM>(E, M, NE, ex, cfgs[i].first, cfgs[i].second, tag, nullptr, toy && i == 0, ord.data());
        ++cases;
    }
}

int main(int argc, char** argv)
{
    std::mt19937_64 g(20261026);
    const std::vector<std::pair<Cfg1, Cfg2>> cfgs = {{{4, 4, 4}, {8, 4}}, {{2, 4, 8}, {4, 8}}, {{2, 6, 4}, {8, 8}}, {{4, 2, 8}, {2, 4}}, {{8, 2, 4}, {8, 4}}};
    long cases = 0;

    // header constants the kernels rely on
    EXPECT(ro_slots_bound(12288, 384) == 384 && ro_slots_bound(100, 384) == 100 && pw_tile_bound(6144, 384) == 1104 && pw_tile_bound(12288, 384) == 1872, "bound constants (W1024 1104 tiles, W2048 1872)");
    EXPECT(ro_ngroups(1, 4) == 1 && ro_ngroups(4, 4) == 1 && ro_ngroups(5, 4) == 2 && ro_ngroups(256, 8) == 32 && ro_group_nt(5, 4, 1) == 1 && ro_group_nt(5, 4, 0) == 4, "group helpers");
    EXPECT(ro_last_live(1) == 1 && ro_last_live(8) == 8 && ro_last_live(9) == 1 && ro_last_live(15) == 7 && ro_last_live(2048) == 8, "ro_last_live");
    // ro_fc1_rec / ro_fc2_rec == the byte offsets k_fc1_mt / k_fc2_mt compute (units of GR)
    {
        std::mt19937_64 r(5);
        long bad = 0;
        for (int i = 0; i < 200000; ++i)
        {
            const int e = int(r() % 384), ch = int(r() % NCH), rec = int(r() % NG1), up = int(r() % 2), ot = int(r() % OT), rc2 = int(r() % NG2);
            const size_t old1 = (size_t(e) * NCH * NG1 * 2 + (size_t(ch) * NG1 + rec) * 2) + up;                 // eb + (chunk * NG + rec) * 2 * GR + up * GR, in GR units
            const size_t old2 = size_t(e) * OT * NG2 + size_t(ot) * NG2 + rc2;                                    // eb + ot * NG * GR + rec * GR
            bad += ro_fc1_rec(e, ch, rec, up, NCH, NG1) != old1;
            bad += ro_fc2_rec(e, ot, rc2, OT, NG2) != old2;
        }
        EXPECT(bad == 0, "ro_fc1_rec / ro_fc2_rec differ from the M<=32 kernels' byte offsets in %ld of 400000 samples", bad);
        printf("T7b0 record offsets: ro_fc1_rec / ro_fc2_rec == k_fc1_mt / k_fc2_mt byte offsets (gate/up, chunk, k record, expert), 400000 samples -> %s\n", bad ? "FAIL" : "OK");
        {   // T7f production geometry: fc1 KS8 CW2 and fc2 W2=8 ranges fit one 8 KiB chunk per lane (the kernels static_assert the same), and the figures in the docs
            const size_t b1 = ro_fc1_pf_recs(2, NG1) * GR_, b2 = ro_fc2_pf_recs(8, NG2) * GR_;
            EXPECT(b1 == 174080 && b2 == 39168 && b1 <= size_t(RO_PF_LANES) * RO_PF_CHUNK && b2 <= size_t(RO_PF_LANES) * RO_PF_CHUNK && RO_PF_CHUNK % 16 == 0, "L2LA production ranges: fc1 %zu B, fc2 %zu B", b1, b2);
            unsigned o = 99, z = 99;                                                    // ro_pf_chunk edges: exact multiple, one short chunk, empty
            EXPECT(ro_pf_chunk(8192, 0, o, z) && o == 0 && z == 8192 && !ro_pf_chunk(8192, 1, o, z), "ro_pf_chunk(8192): one full chunk, then none");
            EXPECT(ro_pf_chunk(8208, 1, o, z) && o == 8192 && z == 16 && !ro_pf_chunk(8208, 2, o, z), "ro_pf_chunk(8208): a full chunk and a 16 B tail");
            EXPECT(!ro_pf_chunk(0, 0, o, z) && ro_pf_chunk(16, 0, o, z) && o == 0 && z == 16, "ro_pf_chunk(0 / 16)");
            printf("T7f L2LA ranges (ro_pf_*): production fc1 CW2 = %zu B (%zu chunks), fc2 W2=8 = %zu B (%zu chunks); per-CTA target / range / byte-chunk checks run inside T7a/T7b/T7c below\n", b1, (b1 + RO_PF_CHUNK - 1) / RO_PF_CHUNK, b2, (b2 + RO_PF_CHUNK - 1) / RO_PF_CHUNK);
        }
    }

    // ---- generated routes
    const int fa = g_fail;
    for (int kind = 0; kind < KINDS; ++kind)
    {
        for (int W : {33, 256, 513, 846, 1024})
        {
            static int ex[1024][8]; static float w[1024][8];
            gen(static_cast<Kind>(kind), g, W, 384, 6, ex, w);
            char what[96]; snprintf(what, sizeof what, "384/6 MM1024 %s W=%d", kname[kind], W);
            run_case<384, 6, 1024>(W, ex, w, what, cfgs, W == 33 || W == 1024, cases);
        }
        for (int W : {1025, 2048})
        {
            static int ex[2048][8]; static float w[2048][8];
            gen(static_cast<Kind>(kind), g, W, 384, 6, ex, w);
            char what[96]; snprintf(what, sizeof what, "384/6 MM2048 %s W=%d", kname[kind], W);
            run_case<384, 6, 2048>(W, ex, w, what, cfgs, W == 2048, cases);
        }
        for (int W : {64, 1024})
        {
            static int ex[1024][8]; static float w[1024][8];
            gen(static_cast<Kind>(kind), g, W, 128, 3, ex, w);
            char what[96]; snprintf(what, sizeof what, "128/3 MM1024 %s W=%d", kname[kind], W);
            run_case<128, 3, 1024>(W, ex, w, what, cfgs, false, cases);
        }
    }
    printf("T7a/T7b/T7c generated routes (5 kinds; 384/6 W 33..2048 incl. 846, 128/3 W 64/1024; 4 (fc1,fc2) config pairs each): %ld checks -> %s\n", cases, g_fail == fa ? "OK" : "FAIL");

    // ---- real routes (one prompt of 1200 rows, layers 0/2/20/39); W=2048 is synthetic (rows of two layers' routes)
    const std::string dir = argc > 1 ? argv[1] : "";
    struct Real { std::vector<int32_t> idx; std::vector<float> w; };
    std::vector<Real> real;
    const int layers[4] = {0, 2, 20, 39};
    for (int L : layers)
    {
        Real r; r.idx.resize(1200 * 6); r.w.resize(1200 * 6);
        if (dir.empty()) break;
        FILE* f = fopen((dir + "/L" + std::to_string(L) + ".router_idx.bin").c_str(), "rb");
        FILE* h = fopen((dir + "/L" + std::to_string(L) + ".router_w.bin").c_str(), "rb");
        if (f && h && fread(r.idx.data(), 4, r.idx.size(), f) == r.idx.size() && fread(r.w.data(), 4, r.w.size(), h) == r.w.size()) real.push_back(r);
        if (f) fclose(f);
        if (h) fclose(h);
    }
    const int fb = g_fail;
    long real_cases = 0;
    const int Ws[5] = {33, 256, 512, 846, 1024};
    for (size_t a = 0; a < real.size(); ++a)
    {
        for (int wi = 0; wi < 5; ++wi)
        {
            const int W = Ws[wi];
            static int ex[1024][8]; static float w[1024][8];
            for (int m = 0; m < W; ++m) for (int k = 0; k < 8; ++k) { ex[m][k] = k < 6 ? real[a].idx[m * 6 + k] : -1; w[m][k] = k < 6 ? real[a].w[m * 6 + k] : 0.f; }
            char what[96]; snprintf(what, sizeof what, "p1200 L%d W=%d", layers[a], W);
            run_case<384, 6, 1024>(W, ex, w, what, cfgs, W == 1024, real_cases);
        }
        // W = 1200 on MM = 2048 (the adapter picks MM by rows); and W = 2048 synthetic
        static int ex[2048][8]; static float w[2048][8];
        for (int m = 0; m < 1200; ++m) for (int k = 0; k < 8; ++k) { ex[m][k] = k < 6 ? real[a].idx[m * 6 + k] : -1; w[m][k] = k < 6 ? real[a].w[m * 6 + k] : 0.f; }
        char what[96]; snprintf(what, sizeof what, "p1200 L%d W=1200 (MM2048)", layers[a]);
        run_case<384, 6, 2048>(1200, ex, w, what, cfgs, false, real_cases);
        const size_t b = (a + 1) % real.size();
        for (int m = 1200; m < 2048; ++m) for (int k = 0; k < 8; ++k) { const int r = (m - 1200 + 37 * int(a)) % 1200; ex[m][k] = k < 6 ? real[b].idx[r * 6 + k] : -1; w[m][k] = k < 6 ? real[b].w[r * 6 + k] : 0.f; }
        snprintf(what, sizeof what, "p1200 L%d+L%d W=2048 (synthetic)", layers[a], layers[b]);
        run_case<384, 6, 2048>(2048, ex, w, what, cfgs, true, real_cases);
    }
    printf("T7a/T7b/T7c real p1200 routes (%zu layers x W 33/256/512/846/1024 on MM1024, W=1200 on MM2048, + synthetic W=2048): %ld checks -> %s\n", real.size(), real_cases, real.empty() ? "SKIPPED (no route dir)" : g_fail == fb ? "OK" : "FAIL");

    // ---- T7d modeled bytes per layer and NTG (real routes; D = derived from the enumerated loops, fc1+fc2 together)
    if (!real.empty())
    {
        printf("T7d modeled weight bytes per layer (D, from the enumerated loops; expert = %.5f MB; DRAM-once = touched x expert; L2->SM = sum of expert x ceil(nt/NTG)):\n", EXPERT_BYTES / 1e6);
        printf("    %-4s %-5s %-8s %-4s %-12s %-14s %-9s %-12s\n", "W", "layer", "touched", "NTG", "DRAM-once MB", "L2->SM MB", "ratio", "DRAM ms@255.8GB/s");
        for (int W : {512, 1024})
            for (size_t a = 0; a < real.size(); ++a)
                for (int ci : {0, 1})                                                  // NTG 4 (cfg 0) and NTG 8 (cfg 1)
                {
                    static int ex[1024][8]; static float w[1024][8];
                    for (int m = 0; m < W; ++m) for (int k = 0; k < 8; ++k) { ex[m][k] = k < 6 ? real[a].idx[m * 6 + k] : -1; w[m][k] = k < 6 ? real[a].w[m * 6 + k] : 0.f; }
                    Model md;
                    char what[64]; snprintf(what, sizeof what, "model W=%d L%d", W, layers[a]);
                    std::vector<std::pair<Cfg1, Cfg2>> one = {cfgs[ci]};
                    long dummy = 0;
                    run_case<384, 6, 1024>(W, ex, w, what, one, false, dummy, &md, 0);
                    printf("    %-4d L%-4d %-8d %-4d %-12.1f %-14.1f %-9.2f %-12.3f\n", W, layers[a], md.touched, cfgs[ci].first.NTG, md.dram / 1e6, md.l2sm / 1e6, md.l2sm / md.dram, md.dram / 255.8e9 * 1e3);
                }
    }
    printf("%s (%d failed checks)\n", g_fail ? "HOST TEST FAIL" : "HOST TEST ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
