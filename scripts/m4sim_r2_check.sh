#!/bin/bash
# m4sim checks of GR8_CP_R2, a session at a smaller R than the one the bindings lay out (s2.cuh S2Model::rp, prefill.cuh
# cp_pf_R, rt_rows.cuh Seq::L), on ONE GPU. Every run is the GR8_TEST_S2_XLIST gate: each sequence runs as one session
# past R on a shuffled list of the pool's page classes.
#
#   MODEL_ARGS="--tp 4 --v2 DIR ..." [M4SIM=./m4sim] [M4SIM_MAIN=main's m4sim] [OUT=dir] scripts/m4sim_r2_check.sh
#
# Checks, stdout byte for byte:
#   r2     GR8_CP_R=BIG GR8_CP_R2=SMALL == GR8_CP_R=SMALL: the session's prefix, prefill staging and every step's path follow
#          SMALL, and the bindings' larger layout changes no byte
#          (control: the GR8_CP_R=BIG run without GR8_CP_R2 must differ)
#   r2rt   the same with the rows tier releasing the tail (GR8_RT_STORE=mem:, GR8_RT_LOCAL)
#   main   (M4SIM_MAIN set) without GR8_CP_R2, M4SIM == M4SIM_MAIN on the xlist gate, with and without the rows tier
# Exit 0 = every check PASS. Every run's stdout / stderr stays in OUT.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
M4SIM=${M4SIM:-$HERE/m4sim}
[ -x "$M4SIM" ] || M4SIM=$HERE/../bin/m4sim
[ -x "$M4SIM" ] || { echo "m4sim not found: set M4SIM"; exit 2; }
[ -n "${MODEL_ARGS:-}" ] || { echo "set MODEL_ARGS (--packs DIR, or the --v2 ... --v2-dense ... release form)"; exit 2; }
OUT=${OUT:-$PWD/m4sim-r2-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
LAYERS=${LAYERS:-0,1,2,3,4,6}
TMO=${TMO:-1800}
BIG=${BIG:-256}                     # GR8_CP_R of the boot
SMALL=${SMALL:-128}                 # the session's R
CAP=${CAP:-2048}
PAGES=${PAGES:-16}                  # the xlist run's pages: prefix, its shard or window below place, the rest above
WIN=${WIN:-4}                       # GR8_RT_LOCAL window pages
COMMON_ENV="GR8_DEC_SHARED_SCRATCH=1 GR8_CP=1 GR8_TEST_ROW_LINES=1"
COMMON_ARGS="--layers $LAYERS --cap $CAP --temp 0 --wide --pfv2 --graph --prefill 64 --chunk 8 --mtp 2 --gen 24"
XLIST="GR8_TEST_S2_XLIST=7,$PAGES"
RT="GR8_RT_STORE=mem: GR8_RT_LOCAL=$WIN"

# a prompt of 600 ids (past both Rs) and one of 200 (past SMALL only); an LCG over 1000..149999
read -r P0 P1 < <(awk 'BEGIN { x = 12345; split("600 200", n, " ")
  for (k = 1; k <= 2; ++k) { s = ""; for (i = 0; i < n[k]; ++i) { x = (x * 1103515245 + 12345) % 2147483648; s = s (i ? "," : "") 1000 + x % 149000 }
    printf "%s%s", (k > 1 ? " " : ""), s }
  print "" }')
[ -n "$P1" ] || { echo "no prompts"; exit 2; }
PROMPTS="--prompt $P0 --prompt $P1"

fails=0
declare -a SUMMARY
note() { SUMMARY+=("$1"); echo "$1"; }
# run NAME BIN "ENV..." ARGS...: one m4sim run, its stdout in out.NAME.txt; a nonzero exit fails
run() {
  local name=$1 bin=$2 env=$3; shift 3
  echo "+ $env $bin $*" > "$OUT/cmd.$name.txt"
  local t0=$SECONDS
  env $env timeout "$TMO" "$bin" $MODEL_ARGS "$@" > "$OUT/out.$name.txt" 2> "$OUT/err.$name.txt"
  local rc=$?
  echo "  $name: rc $rc, $((SECONDS - t0)) s"
  if [ $rc != 0 ]; then note "FAIL run $name: exit $rc (see err.$name.txt)"; fails=$((fails + 1)); return 1; fi
}
# differ A B: the control: runs A and B must not print the same bytes, or the check beside them proves nothing
differ() {
  if [ -s "$OUT/out.$1.txt" ] && ! cmp -s "$OUT/out.$1.txt" "$OUT/out.$2.txt"; then note "PASS differ $1 != $2"
  else note "FAIL differ $1 == $2: the gate cannot tell the two Rs apart"; fails=$((fails + 1)); fi
}
same() {
  if [ -s "$OUT/out.$1.txt" ] && cmp -s "$OUT/out.$1.txt" "$OUT/out.$2.txt"; then note "PASS same $1 == $2"
  else note "FAIL same $1 != $2 (diff out.$1.txt out.$2.txt)"; fails=$((fails + 1)); fi
}

echo "# m4sim $M4SIM, R $BIG with a session at $SMALL, layers $LAYERS, out $OUT"
run r2.ref "$M4SIM" "$COMMON_ENV GR8_CP_R=$SMALL $XLIST" $COMMON_ARGS $PROMPTS
run r2 "$M4SIM" "$COMMON_ENV GR8_CP_R=$BIG GR8_CP_R2=$SMALL $XLIST" $COMMON_ARGS $PROMPTS
same r2.ref r2
run r2.big "$M4SIM" "$COMMON_ENV GR8_CP_R=$BIG $XLIST" $COMMON_ARGS $PROMPTS
differ r2.big r2
run r2rt.ref "$M4SIM" "$COMMON_ENV GR8_CP_R=$SMALL $XLIST $RT" $COMMON_ARGS $PROMPTS
run r2rt "$M4SIM" "$COMMON_ENV GR8_CP_R=$BIG GR8_CP_R2=$SMALL $XLIST $RT" $COMMON_ARGS $PROMPTS
same r2rt.ref r2rt
if [ -n "${M4SIM_MAIN:-}" ]; then
  for b in new main; do
    bin=$M4SIM; [ $b = main ] && bin=$M4SIM_MAIN
    run main.x.$b "$bin" "$COMMON_ENV GR8_CP_R=$BIG $XLIST" $COMMON_ARGS $PROMPTS
    run main.xrt.$b "$bin" "$COMMON_ENV GR8_CP_R=$BIG $XLIST $RT" $COMMON_ARGS $PROMPTS
  done
  same main.x.main main.x.new
  same main.xrt.main main.xrt.new
fi
echo
printf '%s\n' "${SUMMARY[@]}" | grep -c '^PASS' | sed 's/^/PASS checks: /'
printf '%s\n' "${SUMMARY[@]}" | grep '^FAIL' || true
printf '%s\n' "${SUMMARY[@]}" > "$OUT/summary.txt"
if [ $fails = 0 ]; then echo "ALL PASS ($OUT)"; exit 0; fi
echo "$fails FAILED ($OUT)"; exit 1
