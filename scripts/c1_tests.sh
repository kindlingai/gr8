#!/bin/bash
# Build (if needed) and run the unit tests on ONE GPU node (a GB10 Spark with CUDA 13): row_budget_test (dense GEMVs and K1
# against chunks of 8 rows), s2_attn_test (batched attention over several sessions against the per-session calls;
# needs EXTRA=-DDEC_S2, else it only prints SKIP) and dec_heads_test (the decode attention kernels against the ones they
# replaced).
#   scripts/c1_tests.sh <bin_dir> [test args]       default bin_dir = ./bin; TEST=row_budget_test|s2_attn_test runs one
#   G6_NR=4 scripts/c1_tests.sh bin4                the TP4 build of the test
#   EXTRA=-DDEC_S2 scripts/c1_tests.sh bin_s2       the 32-row build: M = 1..32 against chunks of 8 (without it: M <= 8)
#   ARCH=120a ...                                   RTX 50xx instead of the GB10 (default 121a)
# Compiles engine/runtime/tests/<test>.cu with the K1H2 flags of scripts/build.sh into <bin_dir>/<test> (s2_attn_test also
# compiles attn/attn.cu and attn/bind.cu into it), skipping a compile when the binary is newer than the sources and was built
# with the same flags.
# Exit code: 0 = every case PASS, 1 = a case FAILED, 2 = CUDA error, other = build failure (the first failing test's code).
# Test args go to the selected test (TEST=...), with both selected to row_budget_test:
#   row_budget_test: --only gv|k1, --time [reps] (K1 timing over M = 6..24, see scripts/tune_k1_split.sh)
#   s2_attn_test:    --only rep|cp, --seeds N, --no-long (skips the two cases above 32768 tokens)
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$REPO/bin}" && cd "${1:-$REPO/bin}" && pwd)
shift || true
export PATH=/usr/local/cuda/bin:$PATH
A=${ARCH:-121a}
# keep in step with K1H2_FLAGS in scripts/build.sh (DEC_SLOT=2 and DEC_I4PAIR=1 are the paths the engine ships)
K1H2_FLAGS="-DPF_FC2GL=6 -DPF_GEMMF=1 -DPF_XQ8C=1 -DPF_RNWIDE=1 -DPF_GEMMR=1 -DPF_DRAFTKV -DPF_GEMMRF=1 -DPF_QOBH=1 \
-DDEC_SLOT=2 -DDEC_CPB=1 -DDEC_AVP=1 -DPF_QOROPE=1 -DPF_ISGRT=1 -DPF_ROPE2=1 -DDEC_CPB2=64 -DDEC_I4PAIR=1 -DSEL32 \
-DG6_CP=1 -DG6_CP_PF_HASH2=1 -DDEC_IDX31=1"
FLAGS="-gencode=arch=compute_$A,code=sm_$A $K1H2_FLAGS ${G6_NR:+-DG6_NR=$G6_NR} ${EXTRA:-}"
build_run() {   # build_run <test> <extra sources...>  (test args in ARGS)
  local name=$1; shift
  local SRC="$REPO/engine/runtime/tests/$name.cu" BIN="$OUT/$name"
  local stale=0
  [ -x "$BIN" ] || stale=1
  [ "$(cat "$BIN.flags" 2>/dev/null)" = "$FLAGS" ] || stale=1
  if [ "$stale" = 0 ] && [ -n "$(find "$REPO/engine" -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.h' \) -newer "$BIN" -print -quit)" ]; then stale=1; fi
  if [ "$stale" = 1 ]; then
    echo "building $BIN ($FLAGS)"
    rm -f "$BIN" "$BIN.flags"
    # shellcheck disable=SC2086
    nvcc -O2 -std=c++17 $FLAGS -o "$BIN" "$SRC" "$@"
    echo "$FLAGS" > "$BIN.flags"
  else
    echo "up to date: $BIN"
  fi
  local rc=0
  "$BIN" ${ARGS[@]+"${ARGS[@]}"} || rc=$?
  return $rc
}
TEST=${TEST:-all}
ARGS=("$@")
rc=0
if [ "$TEST" = all ] || [ "$TEST" = row_budget_test ]; then
  build_run row_budget_test || rc=$?
fi
if [ "$TEST" = all ] || [ "$TEST" = s2_attn_test ]; then
  [ "$TEST" = all ] && ARGS=()
  case "$FLAGS" in
    *-DDEC_S2*) build_run s2_attn_test "$REPO/engine/attn/attn.cu" "$REPO/engine/attn/bind.cu" -lcublas || { r=$?; [ "$rc" = 0 ] && rc=$r; } ;;
    *) build_run s2_attn_test || { r=$?; [ "$rc" = 0 ] && rc=$r; } ;;
  esac
fi
if [ "$TEST" = all ] || [ "$TEST" = dec_heads_test ]; then
  [ "$TEST" = all ] && ARGS=()
  build_run dec_heads_test "$REPO/engine/attn/bind.cu" -lcublas || { r=$?; [ "$rc" = 0 ] && rc=$r; }
fi
if [ "$rc" = 0 ]; then echo "c1_tests: PASS"; else echo "c1_tests: FAIL (exit $rc)"; fi
exit "$rc"
