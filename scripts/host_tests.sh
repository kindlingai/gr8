#!/bin/bash
# Host-only tests (g++, no CUDA): the S2 batch planner (engine/runtime/s2_batch.h, with a DFlash2 session), the S2 control barrier, the
# pool's page lists and prefix reuse (s2/pool.h), the commit frame (s2/control_protocol.h), and the pack manifest parser
# (engine/loader/g6pack.cpp).
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
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_conc_replay_test.cpp" -o "$T/pool_conc_replay_test"
"$T/pool_conc_replay_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_pages_test.cpp" -o "$T/pool_pages_test"
"$T/pool_pages_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime/s2" "$REPO/engine/runtime/s2/tests/control_protocol_test.cpp" -o "$T/control_protocol_test"
"$T/control_protocol_test"
g++ -std=c++20 -O2 -Wall -pthread -I "$REPO/engine/loader" "$REPO/engine/loader/tests/manifest_test.cpp" \
  "$REPO/engine/loader/g6pack.cpp" -o "$T/manifest_test"
"$T/manifest_test"
