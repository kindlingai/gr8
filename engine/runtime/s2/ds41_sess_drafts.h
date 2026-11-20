// GR8_DS_SESS_DRAFTS=1 (DS41 DSpark serve, --s2-sessions above the one-step drafting cap): the drafts a session verifies in a shared round
// depend on the sessions active in that round, so the main step stays within the row cap. Host only: s2_batch.h (Batch::set_sess_drafts),
// g6m4.cu (the boot checks and the captured main widths) and s2/tests/batch_ds41_sessdrafts_test.cpp read the same rule.
//   kv(n) = min(K, max_rows / n - 1): the most drafts that keep n sessions' verify rows within the row cap. K = 5, max_rows = 32:
//   1-5 -> 5 (6 rows, as without the knob), 6 -> 4 (5 rows), 7-8 -> 3 (4 rows), 9-10 -> 2 (3 rows), 11-16 -> 1 (2 rows), above 16 -> 0
//   (1 row, no draft step). The draft launches do not depend on kv (always 5 rows a session) and the ns-session graphs cover 1..6 rows.
// The DSpark draft step is always 5 rows a session (the block), so one launch holds at most min(max_rows / 5, MARKOV_MAXS) sessions
// (6 at 32 rows): Ds41Round runs a draft step of more sessions as consecutive launches of at most that many (sess_drafts_chunk).
#pragma once
#include <algorithm>
#include <utility>
#include <vector>
#include "../../scheduler-rs/g6sched.h"
namespace s2 {
// the planner's rule (engine/scheduler-rs planner.rs), so the boot checks and the captured widths read the same one
inline int sess_drafts_kv(int K, int max_rows, int n) { return g6s2_sess_drafts_kv(K, max_rows, n); }
// the most sessions that still draft (kv >= 1), at most slots
inline int sess_drafts_max(int K, int max_rows, int slots) { return g6s2_sess_drafts_max(K, max_rows, slots); }
// the sessions one DSpark draft launch holds: block rows each within max_rows, at most maxs (the markov step's session tile)
inline int sess_drafts_chunk(int max_rows, int block, int maxs) { return std::max(1, std::min(max_rows / block, maxs)); }
// the most rows a session has in a main step of ns > 1 sessions (g6m4.cu captures ns x 1..this): 1 without DSpark (on = GR8_DS_MTP > 0),
// kv(ns) + 1 with GR8_DS_SESS_DRAFTS, else K + 1 up to 8 sessions and 1 above
inline int ms_rps(bool on, bool sess_drafts, int K, int max_rows, int ns)
{
    return !on ? 1 : sess_drafts ? sess_drafts_kv(K, max_rows, ns) + 1 : ns <= 8 ? K + 1 : 1;
}
// the rows the per-session index scratch of a DS41 step of several sessions is carved for (Model::ds41_sess_rows): the largest ms_rps,
// which is also the bound g6s2::ds41_session_rows puts on a served step (1 row a session, or a verify of K + 1 rows)
inline int index_sess_rows(bool on, int K) { return on ? K + 1 : 1; }
// the launches of a draft step of S sessions, at most cap each, in step order: (first session, sessions). S <= cap: one launch of all S
inline std::vector<std::pair<int, int>> draft_chunks(int S, int cap)
{
    std::vector<std::pair<int, int>> v;
    for (int i = 0; i < S; i += cap) v.emplace_back(i, std::min(cap, S - i));
    return v;
}
}
