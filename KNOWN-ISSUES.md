# Known issues

Status of the crashes and bugs we are still working out in the multi-session (S2) build (`EXTRA=-DDEC_S2`).
All runs: six DGX Sparks, GPU clock capped at 2200 MHz. "S" = concurrent sessions.

## 1. Multi-session decode fault ("FABRIC" error word) - Fixed (#26)

- **Symptom:** the engine stopped (all ranks rc=3) with `fabric 1627389953` (or `...57`), `attn 0 cp 0`, on a main step
  with 6 rows, after minutes of steady decode with two or more chats. Ranks 1-5 logged
  `control recv failed: ret=-1 errno=11 ... remaining=4 of 4 bytes`.
- **Cause:** each worker's control thread waited for the next ADMIT/CANCEL frame with the socket receive timeout
  (`G6_S2_TIMEOUT`, default 120 s). In steady decode no frames are sent, so after 120 s the first worker to time out
  left the epoch before its next main step. The other ranks then hit the fabric wait timeout (~9.3 s) and raised the
  error word naming that worker (p in the low byte). Rank 0 is the leader and never waits idle, so it logged nothing.
- **Fix:** the event reader waits without a timeout until a frame starts (a partial frame still times out); TCP
  keepalive (10 s + 4 x 5 s) and `TCP_USER_TIMEOUT` 30 s detect a lost peer instead.
- **Checked:** main with `G6_S2_TIMEOUT=20` failed the same way at main step 161; the fix ran 9 waves of S = 2/3/4
  chats (8K prompts, gen 4096) for 50 min, 29,997 main steps, at the default timeout with no fault.

## 2. S=3/4 descriptor race - Fixed

- A batched KV-append descriptor was read while it was still zero, before its upload finished (the memset and the upload ran on different streams).
- After the fix: no fault, and exact outputs at S=2/3/4 in 4 of 4 passes, all six ranks rc=0.

## 3. Test harness: exact-check result looked up in the wrong place - Open (harness only)

- Our test harness reported CRITERIA FAIL (`s2exact:missing`) on runs where every exact-check pass succeeded
  (all sessions identical, S=2/3/4). The harness read the result from the wrong place. This is not an engine bug.

## 4. Prefix-reuse miss near 786K tokens - Open

- A request that should reuse a ~786K-token resident prefix misses and does a full re-prefill.
- Next step: log a reason code for each miss and return the longest valid resident prefix, not all-or-nothing.

## 5. Aggregate decode stays flat past S=4 - Open

- Aggregate decode, tok/s: S1 39.7, S2 53.1, S4 52.1, S6 52.8 (2200 MHz cap).
- We do not know yet if the cause is the scheduler or weight bandwidth.

## 6. First step after prefill - Not a bug

- We measured it. The first decode step after the prompt runs at 0.989x the steady step time (median over 35 requests, min 0.66, max 1.13).
- The first step stays on the CUDA-graph path. No non-graph path appears in the trace.
