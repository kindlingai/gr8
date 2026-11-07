// DS41 bounded replay of a non-final prefill chunk (host-only, no CUDA).
// engine/tests/test_ds41_pf_split.cpp checks every rule below; runtime/s2/ds41_prefill.h sizes the collectives from late_rows().
//
// The layers past the last kv source (geom ids >= geom::LATE_LAYER_START = 21) owe a later row only their window KV, and a decode step reaches
// back WINDOW positions. So a chunk of `rows` rows runs:
//   range A = ops [0, b):  embed .. layer 20 full-row, then ST_PF_COMPACT (the last op of U_CONS_F of the last layer before 21: the fused
//                          consume_ffn + hc_attn(21) has already made layer 21's input for every row)
//   range B = ops [b, e):  layers >= 21 over late_rows(rows, tail) rows = the last `tail` rows of the chunk, moved to rows 0 .. tail-1
// The split keys on the real layer id (Op::l), not the position in a --layers subset.
//
// The dial: GR8_DS_PF_TAIL (read once at boot, ds41_pf_tail()). Unset = geom::TAIL_ROWS (128). 0 = off: the chunk runs the unsplit plan at
// `rows` rows. tail >= rows (e.g. tail = W) runs the split plan with no compaction and no window floor:
// byte-identical to off. 1 .. WINDOW-1 is refused (the first decode step reads the last 128 positions' window KV).
// GR8_DS_PF_L20_TAIL=1 (L1, default off; l20_tail() below, set once at boot): the boundary moves to layer 20 (geom::CAND_SOURCE, the last kv
// source). Layer 20's ST_CA_ROW (compressed-KV row + index-K row + bf16 shadow of every position; it reads only hc_y, which hc_attn(20) in
// U_CONS_F(19) made) is hoisted into range A, right before ST_PF_COMPACT, as an op of U_CONS_F(19) with i = 20. The rest of layer 20
// (attn_proj, index_topk, cand_blocks, attn_core, attn_out, MLP) joins range B: its outputs are layer 20's window ring and per-row lists
// (the top-k and candidate lists of the rows that run them), and layers 21..39 read those only for their own (tail) rows.
// DSpark: ST_DS_TAP (layers 37..39) and ST_DS_MAIN sit in range B, so main_rows() is [0, late_rows): the draft rings get the tail positions,
// which hold the window the first draft round reads (q = N-1 needs N-128 .. N-1; the last chunk ends at N-2).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "ds41_ops.h"
#include "../scheduler-rs/g6plan.h"

namespace ds41
{
namespace br
{
constexpr int TAIL_OFF = 0;
constexpr int TAIL_MAX = 4096;   // the largest GR8_DS_PREFILL_W (ds41_prefill_rowcap); plan_sizes.rs DS41_TAIL_MAX
// GR8_DS_PF_L20_TAIL: the host switch the boot sets once (g6m4.cu, m4sim.cu) and tests set directly. false = the boundary at LATE_LAYER_START.
inline bool& l20_tail() { static bool on = false; return on; }
// the first layer id of range B: 21, or 20 with GR8_DS_PF_L20_TAIL=1. The bool forms are for the fabric plan, which runs
// before the boot sets l20_tail(). parse_tail, tail_error, late_rows and late_start are engine/scheduler-rs
// src/plan_sizes.rs, which the plan calls too.
inline int late_layer_start(bool l20) { return l20 ? geom::CAND_SOURCE : geom::LATE_LAYER_START; }
inline int late_layer_start() { return late_layer_start(l20_tail()); }
static_assert(geom::CAND_SOURCE + 1 == geom::LATE_LAYER_START, "L1 moves the boundary one layer: layer 20 is the last kv source and the candidate source");
// GR8_DS_PF_TAIL: -1 = not a valid value (the caller dies with tail_error())
inline int parse_tail(const char* v) { return g6plan_ds41_parse_tail(v, geom::TAIL_ROWS, geom::WINDOW); }
inline std::string tail_error(const char* v, bool l20)
{
    char* e = g6plan_ds41_tail_error(v, geom::WINDOW, late_layer_start(l20));
    const std::string s(e);
    g6plan_str_free(e);
    return s;
}
inline std::string tail_error(const char* v) { return tail_error(v, l20_tail()); }
// rows of range B for a chunk of `rows` rows
inline int late_rows(int rows, int tail) { return g6plan_ds41_late_rows(rows, tail); }
// rows range B leaves out (the compaction shift): 0 = no compaction
inline int shift_rows(int rows, int tail) { return rows - late_rows(rows, tail); }
// first position i in Model::layers with ids[i] >= late_layer_start(), -1 when there is none or it is position 0 (nothing to run full-row)
inline int late_start(const std::vector<int>& ids, bool l20) { return g6plan_ds41_late_start(ids.data(), ids.size(), late_layer_start(l20)); }
inline int late_start(const std::vector<int>& ids) { return late_start(ids, l20_tail()); }
// A non-final chunk's window floor for range B: positions below it were not written to the layers' rings this chunk and are masked
// (attention core). 0 = no floor (no compaction).
inline int win_floor(int context, int rows, int tail) { return shift_rows(rows, tail) ? context - late_rows(rows, tail) : 0; }

struct Split
{
    std::vector<Op> ops;   // the non-final plan with ST_PF_COMPACT inserted
    size_t b = 0;          // range A = [0, b) (ends with ST_PF_COMPACT), range B = [b, ops.size())
    int i_late = -1;       // first position of range B; -1 = no split (ops = the input, b = ops.size())
    bool on() const { return i_late >= 0; }
};
// An op split_plan hoists out of U_ATTN(i_late) into range A: ST_CA_ROW of the boundary layer (only a kv source has one: layer 20 under L1).
inline bool hoisted(const Op& o, int i_late) { return o.stage == ST_CA_ROW && o.unit == U_ATTN && o.ui == i_late; }
// nf = dsp::pf::nonfinal_plan(...) of ids. No layer >= late_layer_start() (or only such layers) -> no split.
inline Split split_plan(const std::vector<Op>& nf, const std::vector<int>& ids)
{
    Split s;
    s.i_late = late_start(ids);
    if (s.i_late < 0) { s.ops = nf; s.b = nf.size(); return s; }
    for (const Op& o : nf)
    {
        if (s.b == 0 && o.unit == U_ATTN && o.ui >= s.i_late)
        {
            for (const Op& h : nf)   // L1: layer 20's ca row, run on every row, as the last op of U_CONS_F(i_late - 1) before the compaction
                if (hoisted(h, s.i_late)) { Op x = h; x.unit = U_CONS_F; x.ui = s.i_late - 1; s.ops.push_back(x); }
            Op c; c.stage = ST_PF_COMPACT; c.unit = U_CONS_F; c.ui = s.i_late - 1; c.i = s.i_late - 1; c.l = ids[s.i_late - 1];
            s.ops.push_back(c);
            s.b = s.ops.size();
        }
        if (s.b > 0 && hoisted(o, s.i_late)) continue;
        s.ops.push_back(o);
    }
    if (s.b == 0) { s.ops = nf; s.b = nf.size(); s.i_late = -1; }   // the plan has no layer >= i_late (cannot happen for a nonfinal plan of ids)
    return s;
}
// the layer position an op works on for the split rule (hc_attn(i+1) sits in U_CONS_F(i): its ui is i)
inline int unit_layer(const Op& o) { return o.unit == U_ATTN || o.unit == U_CONS_A || o.unit == U_MLP || o.unit == U_CONS_F ? o.ui : -1; }
// "" = ok. nf: the non-final plan split_plan got; s: its answer.
inline std::string check_split(const std::vector<Op>& nf, const Split& s, const std::vector<int>& ids)
{
    if (!s.on())
    {
        for (const Op& o : s.ops) if (o.stage == ST_PF_COMPACT) return "an unsplit plan has ST_PF_COMPACT";
        if (s.ops.size() != nf.size() || s.b != nf.size()) return "an unsplit plan is not the non-final plan";
        return late_start(ids) < 0 ? "" : "the layers have a late layer but the plan is not split";
    }
    const int lls = late_layer_start();
    if (s.i_late != late_start(ids) || s.i_late <= 0 || ids[s.i_late] < lls || ids[s.i_late - 1] >= lls)
        return "the split does not sit at the first layer id >= the late layer start";
    if (s.b < 1 || s.b >= s.ops.size() || s.ops[s.b - 1].stage != ST_PF_COMPACT) return "range A does not end with ST_PF_COMPACT";
    int compacts = 0;
    for (const Op& o : s.ops) compacts += o.stage == ST_PF_COMPACT;
    if (compacts != 1) return "the split plan does not have exactly one ST_PF_COMPACT";
    // the hoisted ops (L1: ST_CA_ROW of layer i_late) sit right before the compaction, as ops of U_CONS_F(i_late - 1) with i = i_late, and only
    // when the boundary layer is a kv source (under L1 it must have one)
    size_t h0 = s.b - 1;
    while (h0 > 0 && s.ops[h0 - 1].stage == ST_CA_ROW && s.ops[h0 - 1].i == s.i_late) --h0;
    const size_t nh = s.b - 1 - h0;
    if (nh != (geom::is_kv_source(ids[s.i_late]) ? 1u : 0u)) return "range A does not end with the boundary layer's ca row (exactly when it is a kv source)";
    for (size_t x = h0; x < s.b - 1; ++x)
    {
        const Op& a = s.ops[x];
        if (a.unit != U_CONS_F || a.ui != s.i_late - 1 || a.l != ids[s.i_late] || a.rc != -1) return "a hoisted ca row is not in U_CONS_F of the last full-row layer";
    }
    // removing the compaction and putting each hoisted op back after ST_ATTN_PROJ of layer i_late (unit U_ATTN(i_late)) gives the non-final
    // plan op for op
    std::vector<Op> back;
    for (size_t x = 0; x < s.ops.size(); ++x)
    {
        if (x >= h0 && x < s.b) continue;
        back.push_back(s.ops[x]);
        if (x >= s.b && s.ops[x].stage == ST_ATTN_PROJ && s.ops[x].i == s.i_late)
            for (size_t y = h0; y < s.b - 1; ++y) { Op o = s.ops[y]; o.unit = U_ATTN; o.ui = s.i_late; back.push_back(o); }
    }
    size_t k = 0;
    for (const Op& a : back)
    {
        if (k >= nf.size()) return "the split plan has an op the non-final plan does not have";
        const Op& o = nf[k++];
        if (a.stage != o.stage || a.unit != o.unit || a.ui != o.ui || a.i != o.i || a.l != o.l || a.src != o.src || a.rc != o.rc)
            return "the split plan dropped or reordered an op of the non-final plan";
    }
    if (k != nf.size()) return "the split plan dropped an op of the non-final plan";
    // range A: only layers before i_late (and the hoisted ca row); range B: only layers >= i_late (ST_DS_MAIN, outside the layers, is the
    // last op of range B)
    for (size_t x = 0; x < s.b; ++x)
    {
        const Op& o = s.ops[x];
        if (unit_layer(o) >= s.i_late) return "range A has an op of a late layer";
        // an op of layer i_late in range A: only its entry (hc_attn(i_late) in U_CONS_F(i_late - 1)) and the hoisted ca row
        const bool entry = o.i == s.i_late && (o.stage == ST_HC_ATTN || o.stage == ST_ENGRAM), hoist = x >= h0 && x < s.b - 1;
        if (o.i >= s.i_late && !entry && !hoist) return "range A has an op of a late layer";
    }
    for (size_t x = s.b; x < s.ops.size(); ++x)
    {
        const Op& o = s.ops[x];
        const int u = unit_layer(o);
        if (u < s.i_late || ids[u] < lls) return "range B has an op outside the late layers";
        // ca row: the compressed / index-K rows of every position (layers >= 20 read them for every row). Candidate blocks are a per-row list
        // (the rows that run them; layers 24..36 read it for the same rows, the compaction moves it when range A made it), so they may sit
        // in range B when their layer is the boundary (L1): their keys are the index-K rows the hoisted ca row wrote for every row.
        if (o.stage == ST_EMBED || o.stage == ST_ENGRAM || o.stage == ST_CA_ROW || (o.stage == ST_CAND_BLOCKS && u != s.i_late))
            return "range B has an op that writes state every row needs (embed / engram / ca row / candidate blocks)";
    }
    const Op& c = s.ops[s.b - 1];
    if (c.unit != U_CONS_F || c.ui != s.i_late - 1 || c.i != s.i_late - 1 || c.rc != -1) return "ST_PF_COMPACT is not in U_CONS_F of the last full-row layer";
    if (h0 < 1 || s.ops[h0 - 1].unit != U_CONS_F || s.ops[h0 - 1].ui != s.i_late - 1 || s.ops[h0 - 1].stage == ST_CA_ROW)
        return "ST_PF_COMPACT does not follow that layer's consume_ffn unit";
    return "";
}
// Late tail on the final chunk only (GR8_DS_PF_LATE_FINAL=1, default off). After a chunk ends at `context`, the readers of the layers
// >= 21 window state are: the MAIN step at N-1 (reads N-WINDOW .. N-2), and a later chunk's range B, which reads down to its window floor, or
// WINDOW-1 rows before its first row when it is not compacted (rows <= tail: only the last chunk, since full chunks of W > tail rows compact).
// So no reader goes below N-1-tail-WINDOW+1, and a chunk with rows_after = (N-1) - context >= tail + WINDOW may skip range B
// (ST_PF_COMPACT, the layers >= 21, their collectives): nothing reads its late positions. Every other chunk runs the split above.
// late_final_valid: the flag needs bounded replay on with full chunks compacted (WINDOW <= tail < W).
inline bool late_final_valid(int tail, int width) { return tail != TAIL_OFF && tail >= geom::WINDOW && tail < width; }
inline bool late_skip(int rows_after, int tail) { return tail != TAIL_OFF && rows_after >= tail + geom::WINDOW; }
// the DSpark rows of range B (dsp.main_lo / main_hi): every row it runs
struct Rows { int lo, hi; };
inline Rows main_rows(int rows, int tail) { return {0, late_rows(rows, tail)}; }
}   // namespace br
}   // namespace ds41
