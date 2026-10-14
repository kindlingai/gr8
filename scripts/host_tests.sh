#!/bin/bash
# Host-only tests (g++, no CUDA): the switchless relay's ring and credit accounting, the S2 batch planner
# (engine/runtime/s2_batch.h, with a DFlash2 session and the insert rule), the S2 control barrier, the pool's page lists
# and prefix reuse (s2/pool.h over the Rust pool in engine/kvman-rs, and pool_diff_test: that pool against the C++ one
# it replaced), runs that grow on demand through Driver::step (driver_grow_test, with insert rules), the pool's block
# store in lock-step with its copies run through memory, a file and a loopback KV node (pool_stash_test), the commit
# frame (s2/control_protocol.h), the pack manifest parser (engine/loader/g6pack.cpp), and the host memory reclaim
# after load (engine/runtime/host_reclaim.h).
#   scripts/host_tests.sh
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/comms" "$REPO/engine/comms/tests/relay_test.cpp" -o "$T/relay_test"
"$T/relay_test"
python3 "$REPO/engine/comms/tests/relay_loop_test.py"
# the Rust page allocator (engine/kvman-rs) that s2/pool.h wraps, built here where cargo is on the PATH
KV=$REPO/engine/kvman-rs/target/release/libg6kvman.a
if command -v cargo > /dev/null 2>&1; then
  cargo build --release --quiet --manifest-path "$REPO/engine/kvman-rs/Cargo.toml"
fi
[ -f "$KV" ] || { echo "no $KV: cargo build --release --manifest-path engine/kvman-rs/Cargo.toml"; exit 1; }
KVLIB=("$KV" -lpthread -ldl -lm)
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_round_test.cpp" -o "$T/batch_round_test"
"$T/batch_round_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_dflash_test.cpp" -o "$T/batch_dflash_test"
"$T/batch_dflash_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_dsel_test.cpp" -o "$T/batch_dsel_test"
"$T/batch_dsel_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_insert_test.cpp" -o "$T/batch_insert_test"
"$T/batch_insert_test"
g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/capture_barrier_test.cpp" -o "$T/capture_barrier_test"
"$T/capture_barrier_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_reuse_test.cpp" "${KVLIB[@]}" -o "$T/pool_reuse_test"
"$T/pool_reuse_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_conc_replay_test.cpp" "${KVLIB[@]}" -o "$T/pool_conc_replay_test"
"$T/pool_conc_replay_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_pages_test.cpp" "${KVLIB[@]}" -o "$T/pool_pages_test"
"$T/pool_pages_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_diff_test.cpp" "${KVLIB[@]}" -o "$T/pool_diff_test"
"$T/pool_diff_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/driver_grow_test.cpp" "${KVLIB[@]}" -o "$T/driver_grow_test"
"$T/driver_grow_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_stash_test.cpp" "${KVLIB[@]}" -o "$T/pool_stash_test"
STORES=("mem:" "ssd:$T/store")
NODE=$REPO/engine/kvman-rs/target/release/kvman-remote
if [ -x "$NODE" ]; then   # a small unlocked node on loopback (RLIMIT_MEMLOCK in a container is usually 8 MiB)
  "$NODE" --gb 0.05 --port 0 --bind 127.0.0.1 --min-free-gb 0 --no-lock --ready "$T/node.json" > "$T/node.log" 2>&1 &
  NODE_PID=$!
  for _ in $(seq 50); do [ -s "$T/node.json" ] && break; sleep 0.1; done
  PORT=$(sed -n 's/.*"port":\([0-9]*\).*/\1/p' "$T/node.json")
  [ -n "$PORT" ] && STORES+=("remote:127.0.0.1:$PORT")
fi
"$T/pool_stash_test" "${STORES[@]}"
[ -n "${NODE_PID:-}" ] && kill "$NODE_PID"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime/s2" "$REPO/engine/runtime/s2/tests/control_protocol_test.cpp" -o "$T/control_protocol_test"
"$T/control_protocol_test"
g++ -std=c++20 -O2 -Wall -pthread -I "$REPO/engine/loader" "$REPO/engine/loader/tests/manifest_test.cpp" \
  "$REPO/engine/loader/g6pack.cpp" -o "$T/manifest_test"
"$T/manifest_test"

g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/test_context_buckets.cpp" -o "$T/test_context_buckets"
"$T/test_context_buckets"

# host memory after load (#4): the /proc parser and malloc_trim (glibc)
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/host_reclaim_test.cpp" -o "$T/host_reclaim_test"
"$T/host_reclaim_test"
