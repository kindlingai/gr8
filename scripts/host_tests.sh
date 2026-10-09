#!/bin/bash
# Host-only tests (g++, no CUDA): the S2 batch planner (engine/runtime/s2_batch.h, with a DFlash2 session), the S2 control barrier and the
# pool's prefix reuse (s2/pool.h).
#   scripts/host_tests.sh
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_round_test.cpp" -o "$T/batch_round_test"
"$T/batch_round_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_dflash_test.cpp" -o "$T/batch_dflash_test"
"$T/batch_dflash_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_dsel_test.cpp" -o "$T/batch_dsel_test"
"$T/batch_dsel_test"
g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/capture_barrier_test.cpp" -o "$T/capture_barrier_test"
"$T/capture_barrier_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_reuse_test.cpp" -o "$T/pool_reuse_test"
"$T/pool_reuse_test"
