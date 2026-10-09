#!/bin/bash
# Host-only tests (g++, no CUDA): the S2 batch planner (engine/runtime/s2_batch.h).
#   scripts/host_tests.sh
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_round_test.cpp" -o "$T/batch_round_test"
"$T/batch_round_test"
g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/capture_barrier_test.cpp" -o "$T/capture_barrier_test"
"$T/capture_barrier_test"
