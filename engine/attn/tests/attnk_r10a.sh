#!/bin/bash
# attnk_r10a.sh [attnk] [log]: the round 10 E1 attention arms (GR8_ATTN_CTX3_SP, GR8_ATTN_POSIDX) on one GB10, against the
# served attention flags (BASE). Per cell: determinism, what must stay the same, the error vs stock, and the hot / cold
# timing vs BASE (attnk.cu, "Round 10"). Nothing else on the GPU; clocks as served. About 15-25 minutes.
B=${1:-./attnk}
LOG=${2:-attnk_r10a.log}
export ATTNK_BASE="SC2=4,TR=2,CTX2=4,CTX3=2,SEL2=16,IDX=16,QT=1,FRONT=1"   # the served attention flags (recipe env)
export ATTNK_E1=1                                                           # arms: base SP=64 SP=128 POS=1 POS=1,FRONT=0 SP=64,POS=1
run() { echo "=== $*" ; "$@"; echo "=== exit $?"; }
{
  run "$B" 2048 1000 3 3                            # with the per-kernel CHAIN table of BASE
  for M in 1 3 8; do
    run "$B" 2048 1000 $M 2
    run "$B" 2048 2048 $M 2                         # the 2048 bucket full: the last row at position 2047
    run "$B" 8192 8000 $M 2                         # POS inert past 2048 (byte-identical), SP active
    run "$B" 32768 32000 $M 2
  done
  run "$B" 4096 2049 3 2                            # boundary: 2049 keys take the 4096 bucket (indexer, top-2048 of 2049)
  run "$B" 2048 1000 12 2                           # c4-sized step
  run env ATTNK_SESS=2 "$B" 2048 1000 3 2               # two sessions in the step: per-row tables and positions
  run env ATTNK_SESS=2 "$B" 2048 2000 8 2
  run env ATTNK_TIES=1 "$B" 2048 1000 3 2               # exact indexer ties (stock's id tie-break vs position order)
  run env ATTNK_QSCALE=400 "$B" 2048 1000 3 2           # most probabilities exactly 0: context_pk3's mixed / skipped batches
  run env ATTNK_NANKEY=500 ATTNK_ARMS="POS=1" "$B" 2048 1000 3 2   # one NaN index key: stock drops it (err 4), POS keeps it
} 2>&1 | tee "$LOG"
echo "--- summary ($LOG)"
grep -E "^(E1|TIME|EXACT|=== exit)" "$LOG" | grep -E "FAIL|DIFFER|BAD|exit [^0]" && echo "SOME CHECK FAILED (above)" || echo "all checks passed"
grep -E "^TIME" "$LOG"
