// GR8_DS_SESS_DRAFTS=1 (DS41 DSpark serve, --s2-sessions above the one-step drafting cap): the drafts a session verifies in a shared round
// depend on the sessions active in that round, so the main step stays within the row cap. Host only: s2_batch.h (Batch::set_sess_drafts),
// g6m4.cu (the boot checks and the captured main widths) and s2/tests/batch_ds41_sessdrafts_test.cpp read the same rule.
//   kv(n) = min(K, max_rows / n - 1), and at most 1 above 8 sessions. K = 5, max_rows = 32: 1-5 -> 5 (6 rows, as without the knob),
//   6 -> 4 (5 rows), 7-8 -> 3 (4 rows), 9-16 -> 1 (2 rows), above 16 -> 0 (1 row, no draft step).
// The DSpark draft step is always 5 rows a session (the block), so one launch holds at most min(max_rows / 5, MARKOV_MAXS) sessions
// (6 at 32 rows): Ds41Round runs a draft step of more sessions as consecutive launches of at most that many (sess_drafts_chunk).
#pragma once
#include <algorithm>
#include <utility>
#include <vector>
namespace s2 {
inline int sess_drafts_kv(int K, int max_rows, int n)
{
    if (K <= 0 || n < 1 || max_rows < 1) return 0;
    int kv = std::min(K, max_rows / n - 1);
    if (n > 8) kv = std::min(kv, 1);
    return std::max(kv, 0);
}
// the most sessions that still draft (kv >= 1), at most slots
inline int sess_drafts_max(int K, int max_rows, int slots)
{
    int n = 0;
    while (n < slots && sess_drafts_kv(K, max_rows, n + 1) >= 1) ++n;
    return n;
}
// the sessions one DSpark draft launch holds: block rows each within max_rows, at most maxs (the markov step's session tile)
inline int sess_drafts_chunk(int max_rows, int block, int maxs) { return std::max(1, std::min(max_rows / block, maxs)); }
// the launches of a draft step of S sessions, at most cap each, in step order: (first session, sessions). S <= cap: one launch of all S
inline std::vector<std::pair<int, int>> draft_chunks(int S, int cap)
{
    std::vector<std::pair<int, int>> v;
    for (int i = 0; i < S; i += cap) v.emplace_back(i, std::min(cap, S - i));
    return v;
}
}
