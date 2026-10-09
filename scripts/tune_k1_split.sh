#!/bin/bash
# Time the K1 expert stream at M = 6..24 rows for the K1 split tables (bind.cuh k1_split: row class 1, <= 3, else) and fuse 0 / 1,
# on ONE GB10 with the 32-row build. One line per point, then the best per M. If a table other than the M = 5 one (class 8)
# wins from some M on, k1_split(M) needs another branch there.
#   scripts/tune_k1_split.sh [bin_dir] [reps]       default bin_dir = ./bin_s2, reps = 50
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
EXTRA="-DDEC_S2 ${EXTRA:-}" exec "$REPO/scripts/c1_tests.sh" "${1:-$REPO/bin_s2}" --time "${2:-50}"
