#!/bin/bash
# One command: oracle vectors from the official Engram/ParallelEngramEmbedding, then the host test of engram_math.h.
#   run_math_test.sh <ref-official dir>       Scratch goes to ${TMPDIR:-/tmp}/engram-math-test; nothing is deleted.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ENG="$HERE/../.."
REF="${1:?ref-official dir}"
T="${TMPDIR:-/tmp}/engram-math-test"; mkdir -p "$T"
python3 "$HERE/../gen_engram_vectors.py" --ref "$REF" --out "$T/vectors_math.bin"
g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off -DG6_MODEL_DS41=1 -I"$ENG" -o "$T/test_engram_math" "$HERE/test_engram_math.cpp"
"$T/test_engram_math" "$T/vectors_math.bin"
