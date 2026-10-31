#!/bin/bash
# One command: the CPU test of the Engram token ring (engram_ring.h) against the reference's positional cache. No GPU, no data.
#   run_ring_test.sh [seed]       Scratch goes to ${TMPDIR:-/tmp}/engram-ring-test; nothing is deleted.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ENG="$HERE/../.."
T="${TMPDIR:-/tmp}/engram-ring-test"; mkdir -p "$T"
g++ -std=c++17 -O2 -Wall -Wextra -I"$ENG/engram" -o "$T/test_ring" "$HERE/test_ring.cpp"
"$T/test_ring" "$@"
