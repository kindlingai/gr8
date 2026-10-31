// gr8 DeepSeek-V4.1-Flash read-once prefill MoE: index functions shared by the kernels (ds_prefill_ro.cuh) and the CPU test
// (tests/ds_ro_host_test.cpp). No CUDA dependency.
//
// Plan = dsr::PlanMT<TK, MM> with plan_ref's meaning (ds_plan_mt.h): slot = one distinct expert (ascending id), slot_nt = ceil(n / 8) tiles (up to MM / 8),
// global tile = slot_tile0 + j, the rank-r token of the expert at tile0 + r / 8, column r % 8. The expert loop (k_fc1_ro / k_fc2_ro) is one CTA group per
// expert that walks the expert's tiles in groups of NTG: tile of (group g, position t) = slot_tile0 + g * NTG + t, ntg = min(NTG, nt - g * NTG).
#pragma once
#include "ds_plan_mt.h"
#include <cstddef>

namespace dsr
{
constexpr int RO_MAX = 4096;              // rows per call on the read-once path

// Distinct-expert counter: the distinct experts per layer per window. One record per prefill MoE layer call, written by
// dsr::k_expert_log (ds_prefill_ro.cuh) into a device ring; the host reads it (ds41moe::elog_read). 32 B per record.
constexpr int ELOG_N = 4096;                                        // ring records (40 layers x ~100 windows)
struct ExpertLogRec { int seq, layer, rows, distinct, maxn, routes, start, pad_; };   // distinct = experts with >= 1 route, maxn = routes of the heaviest expert, start = window first position (ExpertLog::win_start when the call ran)
struct ExpertLog { unsigned cursor; int win_start; unsigned pad_[6]; ExpertLogRec rec[ELOG_N]; };
// Host reference of the kernel's counts: tok_expert [M][8] (row stride 8, first TK used), cnt [NE] scratch.
DSM_HD void elog_stats(const int (*tok_expert)[8], int M, int TK, int NE, int* cnt, int& distinct, int& maxn)
{
    for (int e = 0; e < NE; ++e) cnt[e] = 0;
    for (int r = 0; r < M * TK; ++r) ++cnt[tok_expert[r / TK][r % TK]];
    distinct = 0; maxn = 0;
    for (int e = 0; e < NE; ++e) { distinct += cnt[e] > 0; maxn = cnt[e] > maxn ? cnt[e] : maxn; }
}

DSM_HD int ro_ngroups(int nt, int ntg) { return (nt + ntg - 1) / ntg; }                   // tile groups of an expert with nt tiles
DSM_HD int ro_group_nt(int nt, int ntg, int g) { const int left = nt - g * ntg; return left < ntg ? left : ntg; }
DSM_HD int ro_tile(int tile0, int g, int ntg, int t) { return tile0 + g * ntg + t; }      // global tile of position t in group g
DSM_HD int ro_last_live(int n) { return n - 8 * (mt_ntiles(n) - 1); }                    // live columns (1..8) of the expert's last tile
DSM_HD int ro_slots_bound(int R, int NE) { return R < NE ? R : NE; }                      // distinct experts <= this (R = M * TK routes)

// Launch order of the expert loop (GR8_DS_RO_ORDER=1; k_order_ro, ds_prefill_ro.cuh). The CTA groups of k_fc1_ro / k_fc2_ro run in `pos` order (block / UPS); with
// an order array, group `pos` works slot order[pos] (nullptr = slot pos). Only when a (slot, chunk) CTA starts changes, never what it computes.
// Why: a light expert (few routes) is limited by the DRAM stream of its weights, a heavy one (n > S/c = 22.82 us / 0.649 us =
// 35 routes) by compute, and the two walls add up in expert-id order, because it hands the 48 SMs a random run of neighbours. The order merges the two
// classes in proportion to their work, so the heavy experts are spread over the whole kernel (the heaviest first, no heavy expert late) and the light
// experts' stream runs next to their compute:
//   heavy = nt >= RO_ORD_HEAVY_NT tiles (n >= 33 routes), listed by tiles descending (ties: lower slot first); weight of a heavy expert = nt.
//   light = the rest, listed in slot order; weight of a light expert = 1.
//   key of an expert = (weight of the experts before it in its own list) / (total weight of its list), in [0, 1); order = ascending key, a tie goes to the heavy
//   one (the keys of one class are all different).
constexpr int RO_ORD_HEAVY_NT = 5;
DSM_HD int ro_pos_slot(const int* order, int pos) { return order ? order[pos] : pos; }
// key of slot s as num / den (den > 0) and its class cls (0 heavy, 1 light); nt[0..ns) = tiles per slot.
DSM_HD void ro_ord_key(const int* nt, int ns, int s, int& num, int& den, int& cls)
{
    const bool hs = nt[s] >= RO_ORD_HEAVY_NT;
    cls = hs ? 0 : 1;
    int before = 0, total = 0;
    for (int j = 0; j < ns; ++j)
    {
        const bool hj = nt[j] >= RO_ORD_HEAVY_NT;
        if (hj != hs) continue;
        const int wj = hs ? nt[j] : 1;
        total += wj;
        if (hs ? (nt[j] > nt[s] || (nt[j] == nt[s] && j < s)) : j < s) before += wj;
    }
    num = before; den = total;
}
// does slot a (key na / da, class ca) run before slot b (nb / db, cb)? Total order on distinct slots.
DSM_HD bool ro_ord_before(int na, int da, int ca, int a, int nb, int db, int cb, int b)
{
    const long long x = static_cast<long long>(na) * db, y = static_cast<long long>(nb) * da;
    if (x != y) return x < y;
    if (ca != cb) return ca < cb;
    return a < b;
}

// fc1: grid = slots * UPS, UPS = (I / 16) / CW; warp w of a CTA = chunk cg * CW + w / KS, k part w % KS.
DSM_HD void ro_fc1_work(int block, int warp, int ups, int ks, int cw, int& slot, int& chunk, int& part)
{
    slot = block / ups;
    chunk = (block % ups) * cw + warp / ks;
    part = warp % ks;
}
// First byte (in units of one 16 x 64 record, GR bytes) of the gate (up = 0) or up (up = 1) record `rec` of (expert, chunk): fc1 stream
// [E][I/16][K1/64][2][GR].
DSM_HD size_t ro_fc1_rec(int expert, int chunk, int rec, int up, int nch, int ng)
{
    return ((static_cast<size_t>(expert) * nch + chunk) * ng + rec) * 2 + up;
}
// fc2: grid = slots * UPS, UPS = (NOUT / 16) / WARPS; warp w = output tile (block % UPS) * WARPS + w, all k.
DSM_HD void ro_fc2_work(int block, int warp, int ups, int warps, int& slot, int& ot)
{
    slot = block / ups;
    ot = (block % ups) * warps + warp;
}
// fc2 stream [E][NOUT/16][I/64][GR].
DSM_HD size_t ro_fc2_rec(int expert, int ot, int rec, int nout_tiles, int ng)
{
    return (static_cast<size_t>(expert) * nout_tiles + ot) * ng + rec;
}

// GR8_DS_RO_L2LA=<n> (in-kernel L2 lookahead; k_fc1_ro / k_fc2_ro with LAON): warp 0 of CTA b prefetches into L2 the weights CTA b + la will stream, as one contiguous
// byte range cut into 8 KiB bulk prefetches, a chunk per lane. Hint only: no value, no order and no write changes. The range math lives here so the host test runs the
// same functions the kernels run.
constexpr unsigned RO_PF_CHUNK = 8192;                                                    // bytes per cp.async.bulk.prefetch.L2 (a multiple of 16)
constexpr int RO_PF_LANES = 32;                                                           // chunks per CTA: one per lane of warp 0
// The CTA ahead: bp = block + la. False (no prefetch) when la <= 0 or the position bp / ups is not a launched slot group (pos >= ns = *n_slots_dev).
DSM_HD bool ro_pf_target(int block, int la, int ups, int ns, int& bp)
{
    bp = block + la;
    return la > 0 && bp / ups < ns;
}
// First record (units of GR bytes) and record count of the fc1 weights of CTA (expert, chunk group cgrp): CW chunks x NG k records x gate / up, contiguous in the stream.
DSM_HD size_t ro_fc1_pf_rec(int expert, int cgrp, int cw, int nch, int ng) { return ro_fc1_rec(expert, cgrp * cw, 0, 0, nch, ng); }
DSM_HD size_t ro_fc1_pf_recs(int cw, int ng) { return static_cast<size_t>(cw) * ng * 2; }
// Same for fc2: WARPS output tiles x NG records.
DSM_HD size_t ro_fc2_pf_rec(int expert, int ogrp, int warps, int nout_tiles, int ng) { return ro_fc2_rec(expert, ogrp * warps, 0, nout_tiles, ng); }
DSM_HD size_t ro_fc2_pf_recs(int warps, int ng) { return static_cast<size_t>(warps) * ng; }
// Chunk c (= lane) of a range of `bytes`: byte offset and size; false when the range ends before the chunk.
DSM_HD bool ro_pf_chunk(unsigned bytes, int c, unsigned& off, unsigned& sz)
{
    off = static_cast<unsigned>(c) * RO_PF_CHUNK;
    if (off >= bytes) return false;
    sz = bytes - off < RO_PF_CHUNK ? bytes - off : RO_PF_CHUNK;
    return true;
}
}   // namespace dsr
