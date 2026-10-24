#!/bin/bash
# m4sim check of G6_MTP_LASTROW (rank.cuh launch_mlp_block: a draft step routes only each session's last row) on ONE GPU
# (m4sim simulates every rank). The claim: the layer-78 caches, every draft step's last row (token, log-probability,
# head-input and residual rows) and so every target step, round and generated token are the bytes of the run without it.
#
#   MODEL_ARGS="--packs DIR" [M4SIM=./m4sim] [OUT=dir] [QUICK=1] scripts/mtp_lastrow_check.sh
#
# MODEL_ARGS: as scripts/m4sim_split_check.sh (--packs DIR, or --tp 4 --v2 DIR --v2-slices DIR --v2-widths FILE --v2-dense DIR).
# Runs, each with G6_MTP_LASTROW=0 and =1 and G6_MTP_LASTROW_CHECK=1 (a "check" line after every draft step, m4sim header):
#   mtp   --mtp K decode of two prompts (prompt steps of --chunk 8 rows, so first forms of 1..8 rows), plain and graph
#   s2    --s2 --mtp 2, three sessions sharing steps (first forms of several sessions: each session's last row), plain
#         and graph
#   split the mtp graph run with G6_MTP_SPLIT=1 (the eh_proj gather) and G6_DRAFT_HEAD_I4=1 (not QUICK)
# PASS = stdout of the two runs equal after dropping the first-form "draft" lines (their other rows change by design),
# with at least one check line, and the dropped lines differing somewhere (the flag took effect).
# Exit 0 = every check PASS. Every run's stdout / stderr stays in OUT.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
M4SIM=${M4SIM:-$HERE/m4sim}
[ -x "$M4SIM" ] || M4SIM=$HERE/../bin/m4sim
[ -x "$M4SIM" ] || { echo "m4sim not found: set M4SIM"; exit 2; }
[ -n "${MODEL_ARGS:-}" ] || { echo "set MODEL_ARGS (--packs DIR, or the --v2 ... --v2-dense ... release form)"; exit 2; }
OUT=${OUT:-$PWD/m4sim-mtplast-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
LAYERS=${LAYERS:-0,1,2,3,6}
QUICK=${QUICK:-0}
TMO=${TMO:-1800}
TEMP=${TEMP:-0}
MTP_K=${MTP_K:-3}
GEN=${GEN:-48}
CAP=${CAP:-256}
COMMON_ENV="G6_DEC_SHARED_SCRATCH=1 G6_MTP_LASTROW_CHECK=1 ${BASE_ENV:-G6_CP=0}"
COMMON_ARGS="--layers $LAYERS --cap $CAP --temp $TEMP --wide --pfv2"
GRAPH_ENV="G6_DEC_SPLIT=3"
read -r P0 P1 P2 < <(python3 - <<'EOF'
x = 12345
out = []
for n in (72, 48, 24):
    ids = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) % (1 << 31)
        ids.append(str(1000 + x % 149000))
    out.append(','.join(ids))
print(' '.join(out))
EOF
)
PROMPTS="--prompt $P0 --prompt $P1"
fails=0
run() {   # run NAME "ENV..." ARGS...
  local name=$1 env=$2; shift 2
  echo "+ $env m4sim $*" > "$OUT/cmd.$name.txt"
  local t0=$SECONDS
  env $env timeout "$TMO" "$M4SIM" $MODEL_ARGS "$@" > "$OUT/out.$name.txt" 2> "$OUT/err.$name.txt"
  local rc=$?
  echo "  $name: rc $rc, $((SECONDS - t0)) s"
  [ $rc = 0 ] || { echo "FAIL run $name: exit $rc (see $OUT/err.$name.txt)"; fails=$((fails + 1)); }
}
drop='^(q[0-9]+ )?draft [0-9]'
pair() {  # pair NAME "ENV" ARGS...: flag off and on, compared
  local name=$1 env=$2; shift 2
  run "$name.off" "$env G6_MTP_LASTROW=0" "$@"
  run "$name.on" "$env G6_MTP_LASTROW=1" "$@"
  local a=$OUT/out.$name.off.txt b=$OUT/out.$name.on.txt
  local nchk; nchk=$(grep -c ' check ' "$b")
  if ! diff -q <(grep -Ev "$drop" "$a") <(grep -Ev "$drop" "$b") > /dev/null; then
    echo "FAIL $name: lines differ (diff <(grep -Ev '$drop' $a) <(grep -Ev '$drop' $b))"; fails=$((fails + 1))
  elif [ "$nchk" = 0 ]; then
    echo "FAIL $name: no check lines (G6_MTP_LASTROW_CHECK not in this m4sim?)"; fails=$((fails + 1))
  elif diff -q <(grep -E "$drop" "$a") <(grep -E "$drop" "$b") > /dev/null; then
    echo "FAIL $name: the draft lines did not change at all (the flag had no effect?)"; fails=$((fails + 1))
  else
    echo "PASS $name: $(grep -Evc "$drop" "$b") lines equal ($nchk check lines: caches, last rows), $(grep -Ec "$drop" "$b") draft lines dropped"
  fi
}
pair mtp.plain "$COMMON_ENV" $COMMON_ARGS $PROMPTS --chunk 8 --gen "$GEN" --mtp "$MTP_K"
pair mtp.graph "$COMMON_ENV $GRAPH_ENV" $COMMON_ARGS --graph $PROMPTS --chunk 8 --gen "$GEN" --mtp "$MTP_K"
pair s2.plain "$COMMON_ENV" $COMMON_ARGS $PROMPTS --prompt "$P2" --chunk 8 --gen 24 --mtp 2 --s2
pair s2.graph "$COMMON_ENV $GRAPH_ENV" $COMMON_ARGS --graph $PROMPTS --prompt "$P2" --chunk 8 --gen 24 --mtp 2 --s2
if [ "$QUICK" != 1 ]; then
  pair split.graph "$COMMON_ENV $GRAPH_ENV G6_MTP_SPLIT=1 G6_DRAFT_HEAD_I4=1" $COMMON_ARGS --graph $PROMPTS --chunk 8 --gen "$GEN" --mtp "$MTP_K"
  pair temp.graph "$COMMON_ENV $GRAPH_ENV" $COMMON_ARGS --graph $PROMPTS --chunk 8 --gen "$GEN" --mtp "$MTP_K" --temp 0.8 --seed 77
fi
echo "$([ $fails = 0 ] && echo OK || echo FAILED): $fails failure(s); outputs in $OUT"
exit $([ $fails = 0 ] && echo 0 || echo 1)
