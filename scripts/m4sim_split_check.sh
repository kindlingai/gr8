#!/bin/bash
# m4sim checks of the split replicated GEMVs (G6_QA_SPLIT / G6_IDXQ_SPLIT / G6_MTP_SPLIT, rank.cuh QSplit) on ONE GPU
# (m4sim simulates every rank; the gathers go over its loopback in gather phases), and optional row-count probes.
#
#   MODEL_ARGS="--packs DIR" [M4SIM=./m4sim] [OUT=dir] [QUICK=1] [PARTS="ab cp"] [ROWCHECK=1] scripts/m4sim_split_check.sh
#
# MODEL_ARGS: the model source m4sim takes: --packs DIR (DIR/rank<r>.manifest.json + .pack), or the release form with
# its width (--tp 4 --v2 DIR --v2-slices DIR --v2-widths FILE --v2-dense DIR). Parts (default ab cp):
#   ab      flags off vs every flag combination, plain launches and graph replay, stdout byte for byte (the split's
#           claim: the same bytes as the replicated GEMVs):
#           minv  : --minv sweep (rows 0..B-1 in --chunk 4 steps, then one step of M = 1..32..1 rows from row B), row lines
#           mtp   : --mtp K decode with row lines (prompt steps, first / chain drafts, verify steps of K + 1 rows)
#           s2    : --s2 --mtp 2, three sessions sharing steps (G6_S2_ROWHASH)
#           ROWCHECK=1: also, inside every minv run, row B + j must print the same row line at every M > j. That is
#           row-count invariance, which the engine does not promise; it is a probe, not part of the split's check.
#   chunk   (probe) the same prompts teacher-forced at --chunk C = 1..32 (+ generated rows): every row line (keyed by sequence and
#           position) must equal the --chunk 1 run's: the row's logits / residual do not depend on the step's row count or
#           on the rows beside it. Flags off, and (not QUICK) all flags on.
#   ksweep  (probe) --mtp K = 1..7: the generated tokens equal the plain decode's (same prompts, --gen), flags off and on
#   cp      the minv / mtp A/B again with the CP block active (G6_CP=1, G6_CP_R=0: every step context-parallel)
# Exit 0 = every check PASS. Every run's stdout / stderr stays in OUT (out.<name>.txt / err.<name>.txt).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
M4SIM=${M4SIM:-$HERE/m4sim}
[ -x "$M4SIM" ] || M4SIM=$HERE/../bin/m4sim
[ -x "$M4SIM" ] || { echo "m4sim not found: set M4SIM"; exit 2; }
[ -n "${MODEL_ARGS:-}" ] || { echo "set MODEL_ARGS (--packs DIR, or the --v2 ... --v2-dense ... release form)"; exit 2; }
OUT=${OUT:-$PWD/m4sim-split-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
LAYERS=${LAYERS:-0,1,2,3,4,6}       # l % NR = every rotation up to NR 5 (the gather geometry rotates by layer); dense 0-2,
                                    # MoE 3, 4 and 6; indexer layers 0, 1, 2, 6 and a shared-index one (3). G6_DEC_SPLIT=3 needs more than 3
QUICK=${QUICK:-0}
PARTS=${PARTS:-ab cp}
TMO=${TMO:-1800}                    # seconds per run (a lost gather fails by its own timeout, error word 66xxxxxx)
TEMP=${TEMP:-0}
MINV_B=${MINV_B:-8}
MTP_K=${MTP_K:-3}
GEN=${GEN:-24}
CAP=${CAP:-256}
# the decode settings the TP4 engine serves with (scripts/engine_args.sh) that m4sim takes; G6_DEC_SPLIT needs --graph
COMMON_ENV="G6_DEC_SHARED_SCRATCH=1 ${BASE_ENV:-G6_CP=0}"
COMMON_ARGS="--layers $LAYERS --cap $CAP --temp $TEMP --wide --pfv2"
GRAPH_ENV="G6_DEC_SPLIT=3"

# three fixed prompts of 72, 48 and 24 ids (an LCG over 1000..149999: arbitrary text, the same on every run)
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
declare -a SUMMARY
note() { SUMMARY+=("$1"); echo "$1"; }

# run NAME "ENV..." ARGS...: one m4sim run; its stdout in out.NAME.txt. Fails on a nonzero exit (rank lines differ,
# fabric / attention error word) as well.
run() {
  local name=$1 env=$2; shift 2
  [ -s "$OUT/out.$name.txt" ] && grep -q '^# m4sim: ' "$OUT/out.$name.txt" && return 0   # done in an earlier call
  echo "+ $env m4sim $*" > "$OUT/cmd.$name.txt"
  local t0=$SECONDS
  env $env timeout "$TMO" "$M4SIM" $MODEL_ARGS "$@" > "$OUT/out.$name.txt" 2> "$OUT/err.$name.txt"
  local rc=$?
  echo "  $name: rc $rc, $((SECONDS - t0)) s"
  if [ $rc != 0 ]; then note "FAIL run $name: exit $rc (see err.$name.txt / out.$name.txt)"; fails=$((fails + 1)); return 1; fi
  return 0
}
# same A B: stdout of runs A and B byte for byte
same() {
  if cmp -s "$OUT/out.$1.txt" "$OUT/out.$2.txt"; then note "PASS same $1 == $2"
  else note "FAIL same $1 != $2 (diff out.$1.txt out.$2.txt)"; fails=$((fails + 1)); fi
}
# rows REF RUN..: row lines (key: sequence + position, value: after '|'): constant within every run, and equal to REF's
rows() {
  local o rc
  o=$(python3 - "$OUT" "$@" <<'EOF'
import sys
out, ref, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
def load(name):
    d, bad, n = {}, [], 0
    for ln in open(f'{out}/out.{name}.txt'):
        if not ln.startswith('row q'):
            continue
        head, val = ln.rstrip('\n').split(' | ', 1)
        f = head.split()
        key = (f[1], int(f[3]))          # q<seq>, pos
        n += 1
        if key in d and d[key][0] != val:
            bad.append(f'{key[0]} pos {key[1]}: M {d[key][1]} vs M {f[5]}')
        d.setdefault(key, (val, f[5]))
    return d, bad, n
rd, rbad, rn = load(ref)
ok = True
for name in [ref] + runs:
    d, bad, n = (rd, rbad, rn) if name == ref else load(name)
    if not n:
        print(f'FAIL rows {name}: no row lines'); ok = False; continue
    if bad:
        print(f'FAIL rows {name}: {len(bad)} rows differ between row counts, first {bad[:3]}'); ok = False
    diff = [k for k in d if k in rd and d[k][0] != rd[k][0]]
    miss = [k for k in rd if k not in d]
    if diff:
        print(f'FAIL rows {name} vs {ref}: {len(diff)} of {len(d)} rows differ, first {diff[:3]}'); ok = False
    elif not bad:
        print(f'PASS rows {name}: {len(d)} positions, {n} row lines, constant across row counts and equal to {ref}'
              + (f' ({len(miss)} positions of {ref} not run)' if miss and name != ref else ''))
sys.exit(0 if ok else 1)
EOF
)
  rc=$?
  while IFS= read -r l; do note "$l"; done <<< "$o"
  [ $rc = 0 ] || fails=$((fails + 1))
  return 0
}
# gen REF RUN..: the "generated" lines of RUN equal REF's
gen() {
  local ref=$1; shift
  for r in "$@"; do
    if [ -s "$OUT/out.$r.txt" ] && diff -q <(grep '^generated' "$OUT/out.$ref.txt") <(grep '^generated' "$OUT/out.$r.txt") > /dev/null; then
      note "PASS generated $r == $ref"
    else note "FAIL generated $r != $ref"; fails=$((fails + 1)); fi
  done
}

# the flag combinations: QA_SPLIT x IDXQ_SPLIT (x MTP_SPLIT with --mtp); QUICK: off, all on (1 and 2)
if [ "$QUICK" = 1 ]; then QI="0:0 1:1 2:1"; QIM="0:0:0 1:1:1 2:1:1 0:0:1"
else QI="0:0 1:0 2:0 0:1 1:1 2:1"; QIM=""; for a in 0 1 2; do for b in 0 1; do for c in 0 1; do QIM="$QIM $a:$b:$c"; done; done; done; fi
fenv() { local IFS=:; set -- $1; echo "G6_QA_SPLIT=$1 G6_IDXQ_SPLIT=$2${3:+ G6_MTP_SPLIT=$3}"; }
ftag() { echo "$1" | tr -d :; }

ab_part() {   # $1 = tag prefix, $2 = extra env (cp)
  local pre=$1 xenv=$2 form fe ge g tag
  for form in plain graph; do
    g=""; ge=""; [ $form = graph ] && { g=--graph; ge=$GRAPH_ENV; }
    for f in $QI; do
      tag=${pre}minv.$form.$(ftag "$f")
      run "$tag" "$COMMON_ENV $xenv $ge G6_ROW_LINES=1 $(fenv "$f")" $COMMON_ARGS $g $PROMPTS --chunk 4 --minv "$MINV_B"
    done
    for f in $QIM; do
      tag=${pre}mtp.$form.$(ftag "$f")
      run "$tag" "$COMMON_ENV $xenv $ge G6_ROW_LINES=1 $(fenv "$f")" $COMMON_ARGS $g $PROMPTS --chunk 8 --gen "$GEN" --mtp "$MTP_K"
    done
    [ -n "$xenv" ] && continue        # CP block: no --s2 (S2 runs below R)
    for f in $QIM; do
      tag=${pre}s2.$form.$(ftag "$f")
      run "$tag" "$COMMON_ENV $ge G6_S2_ROWHASH=1 $(fenv "$f")" $COMMON_ARGS $g $PROMPTS --prompt "$P2" --chunk 8 --gen 16 --mtp 2 --s2
    done
  done
  for form in plain graph; do
    for f in $QI; do [ "$f" = 0:0 ] || same "${pre}minv.$form.00" "${pre}minv.$form.$(ftag "$f")"; done
    for f in $QIM; do [ "$f" = 0:0:0 ] || same "${pre}mtp.$form.000" "${pre}mtp.$form.$(ftag "$f")"; done
    [ -n "$xenv" ] || for f in $QIM; do [ "$f" = 0:0:0 ] || same "${pre}s2.$form.000" "${pre}s2.$form.$(ftag "$f")"; done
  done
  # plain launches vs graph replay (the m4sim contract; not split-specific)
  same "${pre}minv.plain.00" "${pre}minv.graph.00"
  same "${pre}mtp.plain.000" "${pre}mtp.graph.000"
  [ -n "$xenv" ] || same "${pre}s2.plain.000" "${pre}s2.graph.000"
  # ROWCHECK=1, inside the minv runs: row B + j the same at every row count (and the chunk-4 prefix rows once each)
  [ "${ROWCHECK:-0}" = 1 ] || return 0
  local all=()
  for form in plain graph; do for f in $QI; do all+=("${pre}minv.$form.$(ftag "$f")"); done; done
  rows "${all[@]}"
}

chunk_part() {
  local cs f tag refs set sets
  # flags:chunks sets. Flags off at every chunk; flags on at a few (the A/B part already ties them to flags off at M=1..32)
  if [ "$QUICK" = 1 ]; then sets="0:0:0=1,2,3,5,8,13,32"; else sets="0:0:0=$(seq -s, 1 32) 1:1:0=1,2,3,5,8,13,32 2:1:0=1,7,16,32"; fi
  for set in $sets; do
    f=${set%%=*}; cs=$(echo "${set#*=}" | tr , ' ')
    refs=()
    for c in $cs; do
      tag=chunk.$(ftag "$f").c$c
      run "$tag" "$COMMON_ENV $GRAPH_ENV G6_ROW_LINES=1 $(fenv "$f")" $COMMON_ARGS --graph $PROMPTS --chunk "$c" --gen "$GEN"
      refs+=("$tag")
    done
    rows "${refs[@]}"
  done
  # the chunk runs against the minv sweep (rows from row B at every M): the same row lines
  [ -s "$OUT/out.minv.graph.00.txt" ] && rows chunk.000.c1 minv.graph.00
}

ksweep_part() {
  local fl f k tag
  if [ "$QUICK" = 1 ]; then fl="0:0:0"; else fl="0:0:0 1:1:1"; fi
  for f in $fl; do
    run "ks.$(ftag "$f").plain" "$COMMON_ENV $GRAPH_ENV $(fenv "$f" | sed 's/ G6_MTP_SPLIT=[01]//')" $COMMON_ARGS --graph $PROMPTS --chunk 8 --gen "$GEN"
    local ks=()
    for k in 1 2 3 4 5 6 7; do
      tag=ks.$(ftag "$f").k$k
      run "$tag" "$COMMON_ENV $GRAPH_ENV $(fenv "$f")" $COMMON_ARGS --graph $PROMPTS --chunk 8 --gen "$GEN" --mtp "$k"
      ks+=("$tag")
    done
    gen "ks.$(ftag "$f").plain" "${ks[@]}"
  done
}

echo "# m4sim $M4SIM, layers $LAYERS, out $OUT, parts: $PARTS, quick $QUICK"
for part in $PARTS; do
  case $part in
    ab) ab_part "" "" ;;
    chunk) chunk_part ;;
    ksweep) ksweep_part ;;
    cp) ab_part "cp." "G6_CP=1 G6_CP_R=0" ;;
    *) echo "unknown part $part"; exit 2 ;;
  esac
done
echo
printf '%s\n' "${SUMMARY[@]}" | grep -c '^PASS' | sed 's/^/PASS checks: /'
printf '%s\n' "${SUMMARY[@]}" | grep '^FAIL' || true
printf '%s\n' "${SUMMARY[@]}" > "$OUT/summary.txt"
if [ $fails = 0 ]; then echo "ALL PASS ($OUT)"; exit 0; fi
echo "$fails FAILED ($OUT)"; exit 1
