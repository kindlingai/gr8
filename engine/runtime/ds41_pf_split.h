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
// DSpark: ST_DS_TAP (layers 37..39) and ST_DS_MAIN sit in range B, so main_rows() is [0, late_rows): the draft rings get the tail positions,
// which hold the window the first draft round reads (q = N-1 needs N-128 .. N-1; the last chunk ends at N-2).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "ds41_ops.h"

namespace ds41
{
namespace br
{
constexpr int TAIL_OFF = 0;
constexpr int TAIL_MAX = 4096;   // the largest GR8_DS_PREFILL_W (ds41_prefill_rowcap)
inline bool tail_valid(int tail) { return tail == TAIL_OFF || (tail >= geom::WINDOW && tail <= TAIL_MAX); }
// GR8_DS_PF_TAIL: -1 = not a valid value (the caller dies with tail_error())
inline int parse_tail(const char* v)
{
    if (!v || !*v) return geom::TAIL_ROWS;
    char* end = nullptr;
    const long t = std::strtol(v, &end, 10);
    if (*end || t < 0 || t > TAIL_MAX || !tail_valid((int)t)) return -1;
    return (int)t;
}
inline std::string tail_error(const char* v)
{
    return std::string("GR8_DS_PF_TAIL=") + (v ? v : "") + ": must be 0 (bounded replay off) or " + std::to_string(geom::WINDOW) + ".." +
           std::to_string(TAIL_MAX) + " (rows of each prefill chunk that run the layers >= " + std::to_string(geom::LATE_LAYER_START) + ")";
}
// rows of range B for a chunk of `rows` rows
inline int late_rows(int rows, int tail) { return tail == TAIL_OFF || tail >= rows ? rows : tail; }
// rows range B leaves out (the compaction shift): 0 = no compaction
inline int shift_rows(int rows, int tail) { return rows - late_rows(rows, tail); }
// first position i in Model::layers with ids[i] >= LATE_LAYER_START, -1 when there is none or it is position 0 (nothing to run full-row)
inline int late_start(const std::vector<int>& ids)
{
    for (int i = 0; i < (int)ids.size(); ++i)
        if (ids[i] >= geom::LATE_LAYER_START) return i > 0 ? i : -1;
    return -1;
}
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
// nf = dsp::pf::nonfinal_plan(...) of ids. No layer >= 21 (or only such layers) -> no split.
inline Split split_plan(const std::vector<Op>& nf, const std::vector<int>& ids)
{
    Split s;
    s.i_late = late_start(ids);
    if (s.i_late < 0) { s.ops = nf; s.b = nf.size(); return s; }
    for (const Op& o : nf)
    {
        if (s.b == 0 && o.unit == U_ATTN && o.ui >= s.i_late)
        {
            Op c; c.stage = ST_PF_COMPACT; c.unit = U_CONS_F; c.ui = s.i_late - 1; c.i = s.i_late - 1; c.l = ids[s.i_late - 1];
            s.ops.push_back(c);
            s.b = s.ops.size();
        }
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
    if (s.i_late != late_start(ids) || s.i_late <= 0 || ids[s.i_late] < geom::LATE_LAYER_START || ids[s.i_late - 1] >= geom::LATE_LAYER_START)
        return "the split does not sit at the first layer id >= LATE_LAYER_START";
    if (s.b < 1 || s.b >= s.ops.size() || s.ops[s.b - 1].stage != ST_PF_COMPACT) return "range A does not end with ST_PF_COMPACT";
    int compacts = 0;
    for (const Op& o : s.ops) compacts += o.stage == ST_PF_COMPACT;
    if (compacts != 1) return "the split plan does not have exactly one ST_PF_COMPACT";
    // removing the compaction gives the non-final plan op for op
    size_t k = 0;
    for (size_t x = 0; x < s.ops.size(); ++x)
    {
        if (x == s.b - 1) continue;
        const Op& a = s.ops[x];
        if (k >= nf.size()) return "the split plan has an op the non-final plan does not have";
        const Op& o = nf[k++];
        if (a.stage != o.stage || a.unit != o.unit || a.ui != o.ui || a.i != o.i || a.l != o.l || a.src != o.src || a.rc != o.rc)
            return "the split plan dropped or reordered an op of the non-final plan";
    }
    if (k != nf.size()) return "the split plan dropped an op of the non-final plan";
    // range A: only layers before i_late; range B: only layers >= i_late (ST_DS_MAIN, outside the layers, is the last op of range B)
    for (size_t x = 0; x < s.b; ++x)
    {
        const int u = unit_layer(s.ops[x]);
        if (u >= s.i_late) return "range A has an op of a layer >= LATE_LAYER_START";
    }
    for (size_t x = s.b; x < s.ops.size(); ++x)
    {
        const Op& o = s.ops[x];
        const int u = unit_layer(o);
        if (u < s.i_late || ids[u] < geom::LATE_LAYER_START) return "range B has an op outside the layers >= LATE_LAYER_START";
        if (o.stage == ST_EMBED || o.stage == ST_ENGRAM || o.stage == ST_CA_ROW || o.stage == ST_CAND_BLOCKS)
            return "range B has an op that writes state every row needs (embed / engram / ca row / candidate blocks)";
    }
    const Op& c = s.ops[s.b - 1];
    if (c.unit != U_CONS_F || c.ui != s.i_late - 1 || c.i != s.i_late - 1 || c.rc != -1) return "ST_PF_COMPACT is not in U_CONS_F of the last full-row layer";
    if (s.b < 2 || s.ops[s.b - 2].unit != U_CONS_F || s.ops[s.b - 2].ui != s.i_late - 1) return "ST_PF_COMPACT does not follow that layer's consume_ffn unit";
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
