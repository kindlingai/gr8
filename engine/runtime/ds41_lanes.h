// DS41 two-lane W prefill (GR8_DS_PF_LANES=2, the GLM pf_step2s port), host-only (no CUDA): the contract the callee (g6m4.cu) is built
// from and engine/tests/test_ds41_lanes.cpp checks. Unset / "1" = today's single lane, byte for byte (nothing here runs).
//
// A quantum of up to 2W rows of ONE session (scheduler.h two_lanes) is two chunks: lane A = the first min(rows, W) rows on the Model m,
// stream st and the MAIN fabric (one id of Dispatch's main sequence); lane B = the rest on a second Model view mB, its own stream and its
// own fabric (its own sequence counter). Both lanes walk the SAME op list (split.ops of one non-final plan).
//
// Shared state: the attention scratch / split / prefill ring are process-wide singletons (attn/ds41_attn_ops.cuh), the Engram engine's
// scratch is one engine, the DSpark main scratch + key rings are one, and the KV / compressed caches are the same session's. So the
// stages that touch them (serial()) run as SECTIONS in one token chain: A(s) before B(s) before A(s') for every later section s'. The
// prologue (h2d copies + the Engram token ring update) is the first link. Everything else of the two lanes may overlap.
// schedule() turns the two lanes' op runs into the host enqueue order with one event per chain link; check() is the happens-before /
// data-flow checker over that order and the per-stage access table below (the buffer class table, as data).
#pragma once
#include "ds41_ops.h"
#include "ds41_pf_split.h"
#include "ds41_prefill_fabric.h"
#include "ds41_rs.h"
#include "ds41_rowshard.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ds41::lanes
{
// GR8_DS_PF_LANES: unset / "" / "1" -> 1, "2" -> 2, anything else -> 0 (the caller dies with kParseError)
inline int parse(const char* v)
{
    if (!v || !*v || std::strcmp(v, "1") == 0) return 1;
    return std::strcmp(v, "2") == 0 ? 2 : 0;
}
constexpr const char* kParseError = "GR8_DS_PF_LANES must be 1 (one lane, the default) or 2 (two-lane W prefill)";

// The rows of one quantum: lane A = the first min(rows, W), lane B = the rest (0 = one lane). rows must be 1..2W.
struct Split { uint32_t a = 0, b = 0; };
inline bool split_ok(uint32_t rows, uint32_t W) { return W >= 1 && rows >= 1 && uint64_t(rows) <= 2 * uint64_t(W); }
inline Split split(uint32_t rows, uint32_t W)
{
    Split s;
    if (!split_ok(rows, W)) return s;
    s.a = std::min(rows, W);
    s.b = rows - s.a;
    return s;
}
// The scheduler's decode cap for two lanes: while a session decodes a quantum is at most W rows (one lane), so a MAIN step never waits
// out a 2W quantum. scheduler.h: decode_prefill_rows (0 = off).
inline uint32_t decode_prefill_rows(int lanes, uint32_t W) { return lanes == 2 ? W : 0; }

// The stages that touch shared scratch or shared caches (the token-chain sections).
inline bool serial(Stage s)
{
    switch (s)
    {
    case ST_ENGRAM: case ST_ATTN_PROJ: case ST_CA_ROW: case ST_INDEX_TOPK: case ST_INDEX_BORROW: case ST_CAND_BLOCKS: case ST_ATTN_CORE:
    case ST_ATTN_OUT: case ST_DS_MAIN:
        return true;
    default:
        return false;
    }
}
// stages that never run in a W prefill chunk (decode / draft / head only)
inline bool chunk_stage(Stage s)
{
    return !(s == ST_STEP_IN || s == ST_FINISH || s == ST_HEAD_LOCAL || s == ST_HEAD_MERGE || s == ST_STEP_OUT || s == ST_DS_DRAFT_IN ||
             s == ST_DS_HEAD_LOCAL || s == ST_DS_MARKOV || s >= ST_COUNT);
}

// One lane's op runs over the shared op list: [b, e) with `rows` rows and the attention window floor of those launches
// (ds41_attn_set_win_floor, a host global read at launch: set before each of the lane's segments).
struct Run { size_t b = 0, e = 0; int rows = 0, win_floor = 0; };
// The callee's runs of one chunk (g6m4.cu dc.prefill): no split = the whole list; skip_late = range A without ST_PF_COMPACT; else
// range A with `rows`, range B with `late` rows and the floor win_floor(context, rows, tail).
inline std::vector<Run> lane_runs(const br::Split& sp, int rows, int late, bool skip_late, int floor_late)
{
    if (!sp.on()) return {Run{0, sp.ops.size(), rows, 0}};
    if (skip_late) return {Run{0, sp.b - 1, rows, 0}};
    return {Run{0, sp.b, rows, 0}, Run{sp.b, sp.ops.size(), late, floor_late}};
}

enum Kind : int { K_PRO, K_OPS, K_SEC, K_JOIN };
// One enqueue of the host order. lane 0 = A (stream st), 1 = B. K_PRO = the lane's prologue (h2d copies, prepare, fabric enqueue_step,
// Engram begin); K_OPS / K_SEC = ds41::run(ops, b, e, rows) (K_SEC: only serial() stages); K_JOIN = lane A waits lane B's end.
// wait: cudaStreamWaitEvent(lane's stream, event[wait]) BEFORE the enqueue; record: cudaEventRecord(event[record], lane's stream) AFTER.
// key: the chain position of a K_PRO (-1) / K_SEC (its first op index); -2 otherwise.
struct Seg { int lane = 0; Kind kind = K_OPS; size_t b = 0, e = 0; int rows = 0, win_floor = 0; long key = -2; int wait = -1, record = -1; };
struct Schedule { std::vector<Seg> seg; int events = 0; };

// lanes[0] = A's runs, lanes[1] = B's (empty = one lane: then the order is A's runs alone, no event).
inline Schedule schedule(const std::vector<Op>& ops, const std::vector<Run> (&lanes)[2])
{
    std::vector<Seg> piece[2];
    for (int L = 0; L < 2; ++L)
    {
        if (lanes[L].empty()) continue;
        Seg p; p.lane = L; p.kind = K_PRO; p.key = -1; p.rows = lanes[L][0].rows;
        piece[L].push_back(p);
        for (const Run& r : lanes[L])
            for (size_t k = r.b; k < r.e;)
            {
                const bool s = serial(ops[k].stage);
                size_t e = k + 1;
                while (e < r.e && serial(ops[e].stage) == s) ++e;
                Seg g; g.lane = L; g.kind = s ? K_SEC : K_OPS; g.b = k; g.e = e; g.rows = r.rows; g.win_floor = r.win_floor;
                g.key = s ? long(k) : -2;
                piece[L].push_back(g);
                k = e;
            }
    }
    struct Link { long key; int lane; size_t at; };
    std::vector<Link> chain;
    for (int L = 0; L < 2; ++L)
        for (size_t i = 0; i < piece[L].size(); ++i)
            if (piece[L][i].kind != K_OPS) chain.push_back({piece[L][i].key, L, i});
    std::stable_sort(chain.begin(), chain.end(), [](const Link& x, const Link& y) { return x.key != y.key ? x.key < y.key : x.lane < y.lane; });
    Schedule o;
    size_t cur[2] = {0, 0};
    long last_at[2] = {-1, -1};   // index in o.seg of each lane's last emitted piece
    int prev_lane = -1; long prev_seg = -1;
    auto emit = [&](int L) { o.seg.push_back(piece[L][cur[L]++]); last_at[L] = long(o.seg.size()) - 1; };
    for (const Link& c : chain)
    {
        while (cur[c.lane] < c.at) emit(c.lane);
        Seg g = piece[c.lane][cur[c.lane]];
        if (prev_lane >= 0 && prev_lane != c.lane)
        {
            if (o.seg[size_t(prev_seg)].record < 0) o.seg[size_t(prev_seg)].record = o.events++;
            g.wait = o.seg[size_t(prev_seg)].record;
        }
        o.seg.push_back(g); ++cur[c.lane]; last_at[c.lane] = long(o.seg.size()) - 1;
        prev_lane = c.lane; prev_seg = last_at[c.lane];
    }
    for (int L = 0; L < 2; ++L) while (cur[L] < piece[L].size()) emit(L);
    if (!piece[1].empty())
    {
        Seg& end_b = o.seg[size_t(last_at[1])];
        if (end_b.record < 0) end_b.record = o.events++;
        Seg j; j.lane = 0; j.kind = K_JOIN; j.wait = end_b.record;
        o.seg.push_back(j);
    }
    return o;
}

// ---------------------------------------------------------------- the buffer class table (data) and the checker
enum Buf : int
{
    BF_IO,        // tok / pos / tgt / sess / cells / stepp (Model)
    BF_TRUNK,     // resid4 + hc_mix / pre / post / comb / done / part (mHC)
    BF_HCY,       // hc_y: the sub-layer input
    BF_PART,      // sc.part[0..1]: this rank's partial before the publish
    BF_MOE,       // MoE RoScratch / PwScratch (per Model: scratch_of)
    BF_FAB,       // fabric send / recv slots, tickets, relay word (per FABRIC)
    BF_IDX_LIST,  // the indexer top-512 list (Ds41IdxModel::topk), read by later layers
    BF_IDX_CAND,  // the layer-20 candidate blocks (Ds41IdxModel scratch), read by the index sources 24..36
    BF_ENG_STEP,  // Engram per-step ids / stage / fork / done / jobs (begin_step -> op)
    BF_DSP_TAP,   // DSpark tap rows (37..39 -> ST_DS_MAIN)
    BF_ATTN_SCR,  // attention scratch / split / prefill ring (process-wide)
    BF_IDX_SCR,   // indexer q_b / wproj / y512 / pre / klen (used inside one index op)
    BF_ENG_SCR,   // Engram engine emb / kv / act-quant codes (inside ST_ENGRAM)
    BF_DSP_SCR,   // DSpark MainS scratch (inside ST_DS_MAIN)
    BF_KV,        // window KV of layer l (per layer)
    BF_CKV,       // compressed / index-K rows of kv source l (per source)
    BF_ENG_RING,  // the slot's Engram token ring
    BF_DSP_RING,  // the draft key rings (136 slots: positions collide)
    BF_COUNT
};
enum Class : int { C_LANE, C_SERIAL, C_CAUSAL };
// LANE: one instance per lane (per fabric for BF_FAB); SERIAL: one shared instance, a lane reads only what it wrote; CAUSAL: one shared
// instance (per layer for KV / CKV) of the session's data, the earlier lane's writes come before every access of the later lane.
struct Table { Class cls[BF_COUNT]; };
inline Table table()
{
    Table t{};
    for (int b = 0; b < BF_COUNT; ++b) t.cls[b] = C_LANE;
    t.cls[BF_ATTN_SCR] = t.cls[BF_IDX_SCR] = t.cls[BF_ENG_SCR] = t.cls[BF_DSP_SCR] = C_SERIAL;
    t.cls[BF_KV] = t.cls[BF_CKV] = t.cls[BF_ENG_RING] = t.cls[BF_DSP_RING] = C_CAUSAL;
    return t;
}
inline const char* buf_name(int b)
{
    static const char* const n[BF_COUNT] = {"io", "trunk", "hc_y", "part", "moe", "fabric", "idx_list", "idx_cand", "engram_step",
                                            "dsp_tap", "attn_scratch", "idx_scratch", "engram_scratch", "dsp_scratch", "kv", "ckv",
                                            "engram_ring", "dsp_ring"};
    return b >= 0 && b < BF_COUNT ? n[b] : "?";
}
// one access of an op, in op order. layer: the per-layer instance (KV / CKV), -1 otherwise.
struct Acc { int buf; bool w; int layer = -1; };
inline std::vector<Acc> access(const Op& o)
{
    const int l = o.l, ks = l >= 0 ? geom::kv_source(l) : -1;
    switch (o.stage)
    {
    case ST_EMBED: return {{BF_IO, false}, {BF_TRUNK, true}};
    case ST_ENGRAM: return {{BF_ENG_STEP, false}, {BF_TRUNK, false}, {BF_ENG_SCR, true}, {BF_ENG_SCR, false}, {BF_FAB, true}, {BF_FAB, false},
                            {BF_TRUNK, true}};
    case ST_HC_ATTN: case ST_HC_FFN: return {{BF_TRUNK, false}, {BF_TRUNK, true}, {BF_HCY, true}};
    case ST_ATTN_PROJ: return {{BF_HCY, false}, {BF_IO, false}, {BF_ATTN_SCR, true}, {BF_KV, true, l}};
    case ST_CA_ROW: return {{BF_ATTN_SCR, false}, {BF_CKV, true, l}};
    case ST_INDEX_TOPK:
    {
        std::vector<Acc> a = {{BF_CKV, false, ks}, {BF_IDX_SCR, true}, {BF_IDX_SCR, false}};
        if (l > 20) a.push_back({BF_IDX_CAND, false});
        a.push_back({BF_IDX_LIST, true});
        return a;
    }
    case ST_INDEX_BORROW: return {{BF_IDX_LIST, true}};
    case ST_CAND_BLOCKS: return {{BF_CKV, false, l}, {BF_IDX_SCR, true}, {BF_IDX_SCR, false}, {BF_IDX_CAND, true}};
    case ST_ATTN_CORE:
    {
        std::vector<Acc> a = {{BF_ATTN_SCR, false}, {BF_KV, false, l}};
        if (ks >= 0) { a.push_back({BF_CKV, false, ks}); a.push_back({BF_IDX_LIST, false}); }
        a.push_back({BF_ATTN_SCR, true});
        return a;
    }
    case ST_ATTN_OUT: return {{BF_ATTN_SCR, false}, {BF_PART, true}, {BF_FAB, true}};   // GR8_DS_PF_DIRECT: the producer writes the send slot
    case ST_PUBLISH_ATTN: case ST_PUBLISH_FFN: return {{BF_PART, false}, {BF_FAB, true}};
    case ST_CONSUME_ATTN: case ST_CONSUME_FFN: return {{BF_FAB, false}, {BF_TRUNK, false}, {BF_TRUNK, true}};
    case ST_ROUTER: return {{BF_HCY, false}, {BF_MOE, true}};
    case ST_SHARED: return {{BF_HCY, false}, {BF_MOE, true}};
    case ST_EXPERTS: return {{BF_HCY, false}, {BF_MOE, false}, {BF_MOE, true}, {BF_PART, true}, {BF_FAB, true}};
    case ST_DS_TAP: return {{BF_TRUNK, false}, {BF_DSP_TAP, true}};
    case ST_DS_MAIN: return {{BF_DSP_TAP, false}, {BF_DSP_SCR, true}, {BF_DSP_SCR, false}, {BF_DSP_RING, true}};
    case ST_PF_COMPACT: return {{BF_TRUNK, false}, {BF_TRUNK, true}, {BF_HCY, true}, {BF_IO, true}, {BF_IDX_LIST, false}, {BF_IDX_LIST, true},
                                {BF_IDX_CAND, false}, {BF_IDX_CAND, true}};
    default: return {};
    }
}
// the prologue: h2d copies, the Engram begin (window over the slot's token ring, then the per-step ids the ST_ENGRAM ops read)
inline std::vector<Acc> prologue_access() { return {{BF_IO, true}, {BF_ENG_RING, false}, {BF_ENG_RING, true}, {BF_ENG_STEP, true}}; }

// What check() needs besides the order: the lanes' runs (to prove coverage), each lane's fabric (0 = main, 1 = lane B's) and first
// position (the lane with the smaller first row is the earlier one for the causal rule).
struct Lanes
{
    std::vector<Run> runs[2];
    int fabric[2] = {0, 1};
    long first[2] = {0, 0};
};
// "" = the order is safe: every lane enqueues exactly its prologue + runs in order, every wait follows its record in host order,
// every pair of accesses of two lanes to one instance with a write is ordered by happens-before, a lane never reads another lane's
// LANE / SERIAL data, and the earlier lane's CAUSAL writes come before every access of the later lane.
inline std::string check(const std::vector<Op>& ops, const Schedule& s, const Lanes& ln, const Table& t)
{
    const size_t n = s.seg.size();
    char m[400];
    // 1. coverage
    for (int L = 0; L < 2; ++L)
    {
        std::vector<const Seg*> mine;
        for (const Seg& g : s.seg) if (g.lane == L && g.kind != K_JOIN) mine.push_back(&g);
        if (ln.runs[L].empty()) { if (!mine.empty()) return "lane " + std::to_string(L) + " has segments but no runs"; continue; }
        if (mine.empty() || mine[0]->kind != K_PRO) return "lane " + std::to_string(L) + " does not start with its prologue";
        size_t r = 0, at = ln.runs[L][0].b;
        for (size_t i = 1; i < mine.size(); ++i)
        {
            const Seg& g = *mine[i];
            if (g.kind == K_PRO) return "lane " + std::to_string(L) + ": two prologues";
            while (r < ln.runs[L].size() && at == ln.runs[L][r].e) { ++r; if (r < ln.runs[L].size()) at = ln.runs[L][r].b; }
            if (r >= ln.runs[L].size()) return "lane " + std::to_string(L) + ": segments past its runs";
            const Run& u = ln.runs[L][r];
            if (g.b != at || g.e <= g.b || g.e > u.e || g.rows != u.rows || g.win_floor != u.win_floor)
            {
                std::snprintf(m, sizeof m, "lane %d: segment [%zu,%zu) rows %d floor %d does not continue run %zu at %zu", L, g.b, g.e, g.rows,
                              g.win_floor, r, at);
                return m;
            }
            for (size_t k = g.b; k < g.e; ++k)
                if (!chunk_stage(ops[k].stage) || serial(ops[k].stage) != (g.kind == K_SEC))
                    return std::string("a segment holds stage ") + stage_name(ops[k].stage) + " of the wrong kind";
            at = g.e;
        }
        while (r < ln.runs[L].size() && at == ln.runs[L][r].e) { ++r; if (r < ln.runs[L].size()) at = ln.runs[L][r].b; }
        if (r != ln.runs[L].size()) return "lane " + std::to_string(L) + " does not run all its ops";
    }
    // 2. events: one record each, every wait after its record in host order
    std::vector<long> rec(size_t(std::max(s.events, 0)), -1);
    for (size_t i = 0; i < n; ++i)
        if (s.seg[i].record >= 0)
        {
            if (s.seg[i].record >= s.events || rec[size_t(s.seg[i].record)] >= 0) return "event recorded twice or out of range";
            rec[size_t(s.seg[i].record)] = long(i);
        }
    for (size_t i = 0; i < n; ++i)
        if (s.seg[i].wait >= 0 && (s.seg[i].wait >= s.events || rec[size_t(s.seg[i].wait)] < 0 || rec[size_t(s.seg[i].wait)] >= long(i)))
        {
            std::snprintf(m, sizeof m, "segment %zu waits event %d before it is recorded (a wait on an unrecorded event orders nothing)", i, s.seg[i].wait);
            return m;
        }
    // 3. happens-before: edges = stream order per lane (K_JOIN is on lane A) + record -> wait. Every edge goes forward in host order,
    //    so walking backwards, reach[i] = the direct successors of i and everything they reach.
    const size_t words = (n + 63) / 64;
    std::vector<std::vector<uint64_t>> edge(n, std::vector<uint64_t>(words, 0)), reach(n, std::vector<uint64_t>(words, 0));
    auto bit = [](const std::vector<uint64_t>& v, size_t b) { return (v[b / 64] >> (b % 64)) & 1; };
    long last[2] = {-1, -1};
    for (size_t i = 0; i < n; ++i)
    {
        const int L = s.seg[i].lane;
        if (last[L] >= 0) edge[size_t(last[L])][i / 64] |= uint64_t(1) << (i % 64);
        if (s.seg[i].wait >= 0) { const size_t r = size_t(rec[size_t(s.seg[i].wait)]); edge[r][i / 64] |= uint64_t(1) << (i % 64); }
        last[L] = long(i);
    }
    for (size_t i = n; i-- > 0;)
        for (size_t j = i + 1; j < n; ++j)
            if (bit(edge[i], j)) { reach[i][j / 64] |= uint64_t(1) << (j % 64); for (size_t w = 0; w < words; ++w) reach[i][w] |= reach[j][w]; }
    auto before = [&](size_t a, size_t b) { return a != b && bit(reach[a], b); };
    // 4. the join: with two lanes the order ends with lane A's K_JOIN after lane B's last segment (the caller fences lane A's stream)
    size_t joins = 0;
    for (const Seg& g : s.seg) joins += g.kind == K_JOIN;
    if (!ln.runs[1].empty())
    {
        long lb = -1;
        for (size_t i = 0; i < n; ++i) if (s.seg[i].lane == 1) lb = long(i);
        if (joins != 1 || s.seg[n - 1].kind != K_JOIN || s.seg[n - 1].lane != 0 || lb < 0 || !before(size_t(lb), n - 1))
            return "two lanes without lane A's join after lane B's last segment";
    }
    else if (joins) return "a join without lane B";
    // 5. accesses
    struct A { size_t seg; size_t seq; int lane; bool w; };
    const int earlier = ln.runs[1].empty() ? 0 : (ln.first[0] <= ln.first[1] ? 0 : 1);
    auto inst = [&](const Acc& a, int L) -> long {
        const Class c = t.cls[a.buf];
        const long lay = a.layer + 1;
        const long who = c == C_LANE ? (a.buf == BF_FAB ? ln.fabric[L] : L) : 2;
        return ((long)a.buf * 64 + lay) * 4 + who;
    };
    std::vector<std::pair<long, A>> all;
    size_t seq = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const Seg& g = s.seg[i];
        auto add = [&](const std::vector<Acc>& v) { for (const Acc& a : v) all.push_back({inst(a, g.lane), A{i, seq++, g.lane, a.w}}); };
        if (g.kind == K_PRO) add(prologue_access());
        else if (g.kind != K_JOIN) for (size_t k = g.b; k < g.e; ++k) add(access(ops[k]));
    }
    std::stable_sort(all.begin(), all.end(), [](const std::pair<long, A>& x, const std::pair<long, A>& y) { return x.first < y.first; });
    for (size_t lo = 0; lo < all.size();)
    {
        size_t hi = lo;
        while (hi < all.size() && all[hi].first == all[lo].first) ++hi;
        const long id = all[lo].first;
        const int buf = int(id / 4 / 64), layer = int((id / 4) % 64) - 1;
        const Class c = t.cls[buf];
        for (size_t x = lo; x < hi; ++x)
            for (size_t y = lo; y < hi; ++y)
            {
                const A& p = all[x].second; const A& q = all[y].second;
                if (p.lane == q.lane || !(p.w || q.w) || x > y) continue;
                if (!before(p.seg, q.seg) && !before(q.seg, p.seg))
                {
                    std::snprintf(m, sizeof m, "race on %s layer %d: lane %d segment %zu and lane %d segment %zu are not ordered", buf_name(buf), layer,
                                  p.lane, p.seg, q.lane, q.seg);
                    return m;
                }
            }
        for (size_t y = lo; y < hi; ++y)
        {
            const A& q = all[y].second;
            if (c == C_CAUSAL)
            {
                if (q.lane == earlier) continue;
                for (size_t x = lo; x < hi; ++x)
                {
                    const A& p = all[x].second;
                    if (p.lane == earlier && p.w && !before(p.seg, q.seg))
                    {
                        std::snprintf(m, sizeof m, "causal %s layer %d: the earlier lane's write (segment %zu) is not before the later lane's access (segment %zu)",
                                      buf_name(buf), layer, p.seg, q.seg);
                        return m;
                    }
                }
                continue;
            }
            if (q.w) continue;
            // the last write before this read: in its own segment (earlier in op order), else the hb-latest earlier write
            long best = -1;
            for (size_t x = lo; x < hi; ++x)
            {
                const A& p = all[x].second;
                if (!p.w) continue;
                const bool prior = p.seg == q.seg ? p.seq < q.seq : before(p.seg, q.seg);
                if (!prior) continue;
                if (best < 0) { best = long(x); continue; }
                const A& b = all[size_t(best)].second;
                if (b.seg == p.seg ? b.seq < p.seq : before(b.seg, p.seg)) best = long(x);
            }
            if (best >= 0 && all[size_t(best)].second.lane != q.lane)
            {
                std::snprintf(m, sizeof m, "lane %d reads %s layer %d (segment %zu) last written by lane %d (segment %zu)", q.lane, buf_name(buf), layer,
                              q.seg, all[size_t(best)].second.lane, all[size_t(best)].second.seg);
                return m;
            }
        }
        lo = hi;
    }
    return "";
}

// ---------------------------------------------------------------- lane B's fabric and sequence
// Lane B's slot table: the main table without the C8-pack late slots (Legate ruling b, 29 Oct): a ragged pack is never a lane-B chunk.
inline pf::FabricRuns fabric_runs_b(int pf_rows, int tail, int i_late, int nl, int nx, int mx, int hid, int nr, size_t head_payload, bool rs_on, bool rowshard = false)
{
    return pf::fabric_runs(pf_rows, tail, i_late, nl, nx, mx, hid, nr, head_payload, rs_on, false, rowshard);
}
struct B12Job {
    bool shard_chunk = false, shard_state = false;
    uint32_t state_bytes = 0;
    rs::Job job;
};
// The same per-chunk B12 fabric contract on lane A and lane B. Ragged/short chunks stay unsharded.
inline B12Job b12_job(const std::vector<uint32_t>& sizes, const std::vector<int>& order,
                     int nc, int rs_k, int rows, int late, int nr, size_t row_bytes,
                     int hid, size_t trunk_bytes, bool rowshard_on, bool y_mxfp8, bool skip_late)
{
    B12Job p;
    p.shard_chunk = rowshard::chunk_on(rowshard_on, rows, late, nr);
    p.shard_state = p.shard_chunk && !skip_late;
    p.state_bytes = p.shard_state ? uint32_t(rowshard::state_bytes(late, hid, trunk_bytes)) : 0;
    p.job = rs::job(sizes, order, nc, rs_k, rows, nr, row_bytes, p.state_bytes, p.shard_chunk);
    if (p.shard_chunk && y_mxfp8)
        for (int q = 0; q < rs_k; ++q) p.job.sizes[size_t(nc + q)] = uint32_t(rowshard::y_mxfp8_bytes(rs::quarter(rows, nr), hid));
    return p;
}
// "" = every collective of one chunk fits the fabric f: the checks of g6m4.cu dc.prefill (payload + tail per slot; with RS the NR slices,
// the AG payloads and, with GR8_DS_PF_DIRECT, the flat slices + tail words). sizes / order = ds41_chunk_collectives; nc = the decode
// collectives; rs_k = the RS collectives (0 off); rows = the chunk rows; mx = MX; nr = NR; row_bytes = HID * 2.
inline std::string fits(const g6::FabricDev& f, const std::vector<uint32_t>& sizes, const std::vector<int>& order, int nc, int rs_k, int rows,
                        int mx, int nr, size_t row_bytes, bool direct)
{
    char m[200];
    for (size_t q = 0; q < sizes.size(); ++q)
        if (sizes[q] + size_t(8) > g6::fab_slot_stride(f, int(q)))
        {
            std::snprintf(m, sizeof m, "collective %zu: %u B + tail does not fit the slot (%zu B)", q, sizes[q], g6::fab_slot_stride(f, int(q)));
            return m;
        }
    if (!rs::on(rs_k > 0, 0, rs_k, rows, mx, nr)) return "";
    if (const char* e = rs::job_error(sizes, nc, rs_k, rows, nr, row_bytes)) return e;
    const rs::Job j = rs::job(sizes, order, nc, rs_k, rows, nr, row_bytes);
    for (int q : j.rs)
    {
        if (j.ss * size_t(nr) > g6::fab_slot_stride(f, q)) return "RS: NR slices do not fit an RS collective's slot";
        if (direct && size_t(rs::quarter(rows, nr)) * row_bytes * nr + 8 * size_t(nr) > g6::fab_slot_stride(f, q))
            return "direct: NR flat slices + tail words do not fit an RS slot";
    }
    for (int q = 0; q < rs_k; ++q)
        if (j.sizes[size_t(nc + q)] + size_t(8) > g6::fab_slot_stride(f, nc + q)) return "RS: the AG payload + tail does not fit its slot";
    return "";
}
// Each fabric's step ids must be consecutive (slot parity = step & 1, doorbell = coll_seq(step, c, ncoll)): lane A's chunk and the MAIN
// steps take Dispatch's main sequence; lane B's chunks take this counter (one per lane-B chunk, never 0). Every rank runs the same
// Driver, so the ids agree.
struct Counter
{
    uint64_t next = 1;
    uint64_t take() { return next++; }
};
}  // namespace ds41::lanes
