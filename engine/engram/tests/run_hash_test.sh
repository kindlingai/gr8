#!/bin/bash
# One command: regenerate constants + cmap + official vectors, check hash_consts.h is unchanged, build and run the CPU hash test.
#   run_hash_test.sh <ref-official dir> <tokenizer.json>
# Scratch files go to ${TMPDIR:-/tmp}/engram-hash-test; nothing is deleted.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ENG="$HERE/../.."
REF="${1:?ref-official dir}"; TOK="${2:?tokenizer.json}"
T="${TMPDIR:-/tmp}/engram-hash-test"; mkdir -p "$T"
python3 "$HERE/../gen_engram.py" --ref "$REF" --tok "$TOK" --out-h "$T/hash_consts.h" --out-cmap "$T/ds41.engram.cmap" --vectors "$T/vectors.bin"
cmp "$T/hash_consts.h" "$HERE/../hash_consts.h" && echo "hash_consts.h == regenerated"
g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -I"$ENG" -o "$T/test_hash" "$HERE/test_hash.cpp"
"$T/test_hash" "$T/ds41.engram.cmap" "$T/vectors.bin"
