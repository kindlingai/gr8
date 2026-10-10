#pragma once
// CPU-only state for g6m4's opt-in, one-conversation prefix cache (--reuse-prefix). Ports K2 13e0825 + 18318ee to the
// K1/CP lineage with one change: the retained prefix is the longest common prefix (LCP) of the new prompt and the
// previous sequence (prompt + ACTUALLY emitted ids), not only a strict append (strict = G6_REUSE_STRICT=1, the K2 rule).
// Ends are exclusive absolute positions. First-form draft rows alone are valid; chain writes use speculative hidden
// states and must never advance draft_end. Main row j depends on tokens 0..j; the MTP layer's row j also on token j + 1,
// so with MTP the retained prefix stops one row before the first differing token.
// Hit = skip > 0 and (noend or prompt.size() >= written_end): every row the old sequence wrote past the new context is
// rewritten by this request before any step reads it (kernels read only positions <= the row's own position; the rule
// is redundant under that, G6_REUSE_NOEND=1 drops it). Miss = reset + full prefill (the switch-off path).
// The grid (G6_REUSE_ALIGN=1, g6m4 sets grid = the cold prefill grid: --prefill MP, one sub-batch; the lane pairing does
// not change a row): a hit retains
// only [0, floor(min(skip, pf_end) / grid) * grid), pf_end = the rows prefilled contiguously from 0 in the current sequence
// (decode-written rows never count). The re-prefill then starts on a cold grid point, and every retained grid cell was one
// complete cold-shaped prefill cell of an earlier request, so a hit computes every row with the cold chunk shape (the
// hit's own suffix chunk shape was otherwise the only difference). Retained < grid -> miss 'grid'. One exception: when the new
// prompt's prefill ends exactly where the retained one did (limit == pf_end, the same prompt sent again), its last cell
// [floor(pf_end / grid) * grid, pf_end) has the retained cell's rows and shape, so all pf_end rows are kept.
#include <algorithm>
#include <cassert>
#include <vector>

namespace g6rt {
struct PrefixReuse
{
    bool strict = false, noend = false;   // G6_REUSE_STRICT=1 / G6_REUSE_NOEND=1 (g6m4 sets them once)
    int grid = 0;                       // G6_REUSE_ALIGN=1 -> the cold prefill grid (rows); 0 = off (the old rule)
    int pf_end = 0;                     // rows [0, pf_end) of the current sequence were written by prefill, contiguously
    int raw = 0;                        // the last begin()'s skip before the grid rule (log only)
    std::vector<int> history;           // previous full prompt + ACTUALLY emitted outputs
    int valid = 0, main_end = 0, draft_end = 0;
    int written_end = 0;                // max row (exclusive) written in any cache of the current sequence, chain rows included
    int dirty_end = 0;                  // max row (exclusive) written in any cache since the last real reset (a miss), over
                                        // every sequence of a NOEND hit chain: every stale row lies in [skip, dirty_end) (2b G2)
    int lcp = 0;                        // of the last begin(), for the log line
    const char* why = "first";          // last begin(): "hit", or the miss reason first / strict / no-lcp / short / grid

    static int common_prefix(const std::vector<int>& a, const std::vector<int>& b)
    {
        const size_t n = std::min(a.size(), b.size());
        return (int)(std::mismatch(a.begin(), a.begin() + n, b.begin()).first - a.begin());
    }
    // limit: rows the prefill path may skip (g6m4: p.first, >= 1 row always goes through the normal path)
    int begin(const std::vector<int>& prompt, int limit, bool mtp)
    {
        lcp = common_prefix(history, prompt);
        int skip = 0;
        if (history.empty()) why = "first";
        else if (strict && !(prompt.size() > history.size() && lcp == (int)history.size())) why = "strict";
        else
        {
            skip = std::min({valid, mtp ? lcp - 1 : lcp, limit});
            if (skip <= 0) { skip = 0; why = "no-lcp"; }
            else if (!noend && (int)prompt.size() < written_end) { skip = 0; why = "short"; }
            else why = "hit";
        }
        raw = skip;
        if (skip && grid > 0)
        {
            skip = skip >= pf_end && limit == pf_end ? pf_end : std::min(skip, pf_end) / grid * grid;
            if (!skip) why = "grid";
        }
        if (!skip) dirty_end = 0;       // a miss resets every cache (g6m4: reset(m) + reset(D.d))
        main_end = draft_end = written_end = pf_end = skip;
        return skip;
    }
    void clear() { history.clear(); valid = main_end = draft_end = written_end = dirty_end = pf_end = raw = 0; }
    // S2: a session that continues a retained run whose begin() gave `skip` (the run's other state stays with the pool)
    void resume(int skip)
    {
        clear();
        main_end = draft_end = written_end = dirty_end = pf_end = skip;
    }
    static void written(int& end, int first, int count)
    {
        assert(first >= 0 && first <= end && count > 0);   // no holes in the retained prefix
        end = std::max(end, first + count);
    }
    void main_rows(int first, int count)
    {
        written(main_end, first, count);
        written_end = std::max(written_end, first + count);
        dirty_end = std::max(dirty_end, written_end);
    }
    // one prefill sub-batch (both lanes): main rows, and the MTP layer's first-form rows
    void prefill_rows(int first, int count, bool mtp)
    {
        if (first == pf_end) pf_end = first + count;   // contiguous from 0 (g6m4 prefills in order from skip)
        main_rows(first, count);
        if (mtp) draft_rows(first, count);
    }
    void draft_rows(int first, int count, bool chain = false)
    {
        if (!chain) written(draft_end, first, count);
        written_end = std::max(written_end, first + count);
        dirty_end = std::max(dirty_end, written_end);
    }
    void finish(const std::vector<int>& prompt, const std::vector<int>& gen, int emitted, bool mtp)
    {
        assert(emitted >= 0 && emitted <= (int)gen.size());
        history = prompt;
        history.insert(history.end(), gen.begin(), gen.begin() + emitted);
        // Last token has no trusted next-token-conditioned draft row. This also
        // discards verify/chain lookahead beyond a stop in the middle of a batch.
        valid = std::min(main_end, std::max(0, (int)history.size() - 1));
        if (mtp) valid = std::min(valid, draft_end);
    }
};
} // namespace g6rt
