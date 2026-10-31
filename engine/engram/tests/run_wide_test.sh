#!/bin/bash
# One command: the host test of Engram at W-row prefill (G1 window, G2 hash, G3 fetch + dequant + 4-rank sum, G4 gate, G5 lookahead),
# then the activation quantizer of the W-row wkv (test_act_quant.cpp, needs the CUDA headers for cuda_fp8.h). No GPU.
# run_hash_test.sh checks the hash against the official vectors.
#   run_wide_test.sh [seed]     Scratch goes to ${TMPDIR:-/tmp}/engram-wide-test; nothing is deleted.
# Exit code: 0 only if both parts passed.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ENG="$HERE/../.."
T="${TMPDIR:-/tmp}/engram-wide-test"; mkdir -p "$T"
g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -pthread -I"$ENG/engram" -o "$T/test_engram_wide" "$HERE/test_engram_wide.cpp" || { echo "COMPILE FAILED"; exit 2; }
rc=0
"$T/test_engram_wide" "$T/rs" "$@" || rc=$?
echo "wide test exit code: $rc"
CUDA_INC="${CUDA_INC:-/usr/local/cuda/include}"
g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -I"$ENG/engram" -I"$CUDA_INC" -o "$T/test_act_quant" "$HERE/test_act_quant.cpp" || { echo "COMPILE FAILED (act_quant)"; exit 2; }
arc=0
"$T/test_act_quant" "$@" || arc=$?
echo "act_quant test exit code: $arc"
[ "$rc" = 0 ] && [ "$arc" = 0 ]
