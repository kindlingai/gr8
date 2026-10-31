// DSpark during PREFILL: the host-side contract of ST_DS_TAP / ST_DS_MAIN for a non-final prompt chunk of W rows (no CUDA).
// docs/DEEPSEEK-DSPARK.md section 9 is the card text; engine/tests/test_ds41_dspark_prefill.cpp checks every rule below.
//
// What the draft block reads of the main model: the key ring of each of its 3 layers, positions q-127 .. q (main keys, written by ST_DS_MAIN
// from the taps of the verify rows) and q+1 .. q+5 (its own block keys). After a prompt of N tokens the first round has q = N-1, so the
// rings must hold the main keys of positions N-128 .. N-1 and nothing older is ever read (the window is 128, q only grows: there is no
// prefix reuse or rewind in serve v1; OQ1 of the card says what a reuse hit must redo).
//
// Prefill quanta (B9 memo b9-prefill-admission.md (a)): the hook runs positions 0 .. N-2 in chunks of W rows (the last one partial) and the
// prompt's LAST token N-1 is a normal MAIN step (PromptTail, with its own head). So EVERY hook chunk is "non-final": no ST_FINISH, no head
// stage, no step_out, collectives 0..79 only (c = 80 is the head merge, never enqueued); ST_DS_TAP and ST_DS_MAIN stay.
//
// Three rules, each with a falsifier in the test:
//   1. nonfinal_plan(): the decode plan minus step_in / finish / head / step_out (and minus the DSpark taps and ST_DS_MAIN when no row of the chunk
//      is kept). Every other op, and every publish (c = 0..79), is untouched.
//   2. main_rows() / keep policy: rows of the chunk that go through ST_DS_MAIN. GR8_DS_PREFILL_KEEP = 0 (default) keeps EVERY row (no pruning,
//      the ring is the same as a decode-step prompt gives). keep >= 128 keeps the last `keep` prompt positions only; below 128 is refused
//      at boot (the first round's window would read slots nobody wrote).
//   3. subchunks(): ST_DS_MAIN runs at most `cap` rows per launch, cap <= min(scratch rows, RING). ONE launch must not hold two rows whose
//      positions differ by a multiple of RING (136): k_kv_prep_pos blocks run in any order, so the surviving key would be a race. Launches
//      in stream order, positions ascending, give "newest position wins" per slot.
#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include "ds41_dspark.h"

namespace ds41
{
namespace dsp
{
namespace pf
{
constexpr int KEEP_ALL = 0;                              // GR8_DS_PREFILL_KEEP default: every prompt row goes through ST_DS_MAIN
constexpr int MS_ROWS = PREFILL_MS_ROWS;                // MainS rows of a prefill-capable build (ds41dsp_ops::build with tap_rows > MX): 128
constexpr int KEEP_MIN = geom::WINDOW;                   // 128: the draft window of the first round (q = N-1) is positions N-128 .. N-1

inline bool keep_valid(int keep) { return keep == KEEP_ALL || keep >= KEEP_MIN; }
// first prompt position whose main key is written, for a prompt of N tokens (positions 0 .. N-1, N-1 = the PromptTail MAIN step)
inline int keep_first_pos(int N, int keep) { return keep == KEEP_ALL ? 0 : std::max(0, N - keep); }

// the chunks of one hook quantum: prompt positions first .. first+rows-1 in chunks of W rows, the last one partial
struct Chunk { int first, rows; };
inline std::vector<Chunk> chunks(int first, int rows, int W)
{
    std::vector<Chunk> c;
    for (int r0 = 0; r0 < rows; r0 += W) c.push_back({first + r0, std::min(W, rows - r0)});
    return c;
}

// Rows [lo, hi) of a chunk (first, rows) that go through ST_DS_TAP / ST_DS_MAIN: the suffix whose positions are >= keep_first_pos. Empty
// (lo == hi) when the whole chunk is older than the kept window; such a chunk runs the plan WITHOUT the DSpark stages.
struct Range { int lo, hi; bool empty() const { return lo >= hi; } };
inline Range main_rows(int first, int rows, int N, int keep)
{
    const int p0 = keep_first_pos(N, keep);
    const int lo = std::min(rows, std::max(0, p0 - first));
    return {lo, rows};
}

// ST_DS_MAIN launches of rows [lo, hi): <= cap rows each, ascending. cap = min(MainS rows, RING).
struct Sub { int row0, rows; };
inline std::vector<Sub> subchunks(Range r, int cap)
{
    std::vector<Sub> s;
    if (cap < 1) return s;
    for (int a = r.lo; a < r.hi; a += cap) s.push_back({a, std::min(cap, r.hi - a)});
    return s;
}
// the largest legal cap: the scratch rows and the ring (two rows of one launch must not share a slot)
inline int cap_for(int scratch_rows) { return std::min(scratch_rows, RING); }

// ---- the plan of a non-final chunk ----------------------------------------------------------------------------------
inline bool is_head_stage(Stage s) { return s == ST_FINISH || s == ST_HEAD_LOCAL || s == ST_HEAD_MERGE || s == ST_STEP_OUT; }
// ST_STEP_IN is dropped too (B1, 8:1x AM PT): launch_step_in = k_step_in(StepCtl{m.in, m.stepp, m.tok, m.pos, m.tgt, .., m.cells, ..}) + s2_launch_cells
// (rank.cuh launch_step_in) writes the first MX rows of tok / pos / tgt, which the W-row prefill callee H2D-copies directly (rank.cuh StepIn tok/pos/tgt are [MX]).
// ST_EMBED (the prologue) stays. Whoever drops the StepIn owns what it did besides the rows (the fabric step word and the S2 cells): B9's s2/ds41_prefill.h.
inline bool is_dropped_stage(Stage s) { return is_head_stage(s) || s == ST_STEP_IN; }
// verify = ds41::plan(ids, 0, nl, true, true, engram, /*dspark=*/true). dsp_rows = false: the chunk has no kept row (no tap, no main-cache update).
inline std::vector<Op> nonfinal_plan(const std::vector<Op>& verify, bool dsp_rows)
{
    std::vector<Op> p;
    for (const Op& o : verify)
    {
        if (is_dropped_stage(o.stage)) continue;
        if (!dsp_rows && (o.stage == ST_DS_TAP || o.stage == ST_DS_MAIN)) continue;
        p.push_back(o);
    }
    return p;
}
inline int count_stage(const std::vector<Op>& p, Stage s) { int n = 0; for (const Op& o : p) n += o.stage == s; return n; }
// the number of fabric publishes of a plan: ST_PUBLISH_ATTN / ST_PUBLISH_FFN take c = 2i / 2i + 1, ST_HEAD_MERGE takes c = 2*nl
inline int publishes(const std::vector<Op>& p) { return count_stage(p, ST_PUBLISH_ATTN) + count_stage(p, ST_PUBLISH_FFN) + count_stage(p, ST_HEAD_MERGE); }

// "" = ok. nf = nonfinal_plan(verify, dsp_rows) as the hook will run it; verify = the decode plan it came from.
inline std::string check_nonfinal_plan(const std::vector<Op>& verify, const std::vector<Op>& nf, bool dsp_rows)
{
    char b[200];
    for (const Op& o : nf)
        if (is_dropped_stage(o.stage) || o.stage == ST_DS_DRAFT_IN || o.stage == ST_DS_HEAD_LOCAL || o.stage == ST_DS_MARKOV)
            return "a non-final chunk plan has a step_in / finish / head / step_out / draft stage";
    if (nf.empty() || nf[0].stage != ST_EMBED) return "a non-final chunk plan must open with ST_EMBED (the prologue stays, ST_STEP_IN does not)";
    // everything else of the decode plan is there, in order (a subsequence, nothing reordered, nothing invented)
    size_t k = 0;
    for (const Op& o : verify)
    {
        if (is_dropped_stage(o.stage) || (!dsp_rows && (o.stage == ST_DS_TAP || o.stage == ST_DS_MAIN))) continue;
        if (k >= nf.size() || nf[k].stage != o.stage || nf[k].i != o.i || nf[k].unit != o.unit) return "a non-final chunk plan dropped or reordered an op of the decode plan";
        ++k;
    }
    if (k != nf.size()) return "a non-final chunk plan has an op the decode plan does not have";
    // collectives 0 .. 2*nl-1 are all published; the head merge (c = 2*nl) is not
    if (count_stage(nf, ST_HEAD_MERGE) != 0) return "a non-final chunk plan publishes the head merge";
    if (publishes(nf) != publishes(verify) - count_stage(verify, ST_HEAD_MERGE)) return "a non-final chunk plan lost a layer publish (c = 0 .. 2*nl-1)";
    if (!dsp_rows)
        return count_stage(nf, ST_DS_TAP) == 0 && count_stage(nf, ST_DS_MAIN) == 0 ? "" : "a chunk with no kept row still has DSpark stages";
    if (count_stage(nf, ST_DS_MAIN) != 1) return "a non-final chunk plan does not have exactly one ST_DS_MAIN";
    int taps = 0, main_at = -1, last_tap = -1, last_pub = -1;
    for (size_t x = 0; x < nf.size(); ++x)
    {
        const Op& o = nf[x];
        if (o.stage == ST_DS_TAP)
        {
            ++taps; last_tap = (int)x;
            if (!is_tap_layer(o.l) || o.unit != U_ATTN || x + 1 >= nf.size() || nf[x + 1].stage != ST_ATTN_PROJ || nf[x + 1].i != o.i)
            { std::snprintf(b, sizeof b, "ST_DS_TAP of layer %d is not directly before attn_proj of the same layer", o.l); return b; }
        }
        if (o.stage == ST_DS_MAIN) main_at = (int)x;
        if (o.stage == ST_PUBLISH_FFN) last_pub = (int)x;
    }
    if (taps != TAPS) return "a non-final chunk plan does not tap layers 37, 38 and 39 exactly once each";
    // ST_DS_MAIN reads the three taps, so it follows the last of them; it sits where finish was: after the last layer's publish_ffn, last op
    if (main_at < 0 || main_at < last_tap || main_at < last_pub || main_at != (int)nf.size() - 1) return "ST_DS_MAIN is not the last op, after the last tap and publish_ffn";
    return "";
}

// ---- memory of the prefill tap buffer ---------------------------------------------------------------------------------
// dsp.tap holds one chunk's rows: [rows][3][5120] BF16. The decode build holds MX rows; a W-row chunk needs W (31,457,280 B at W = 1024).
constexpr size_t tap_bytes(int rows) { return (size_t)rows * TAP_ROW * 2; }
}   // namespace pf
}   // namespace dsp
}   // namespace ds41
