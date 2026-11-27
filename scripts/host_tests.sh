#!/bin/bash
# Host-only tests (g++ and cargo, no CUDA) of the engine's CPU-side logic: the build's object cache, steering contracts,
# the fabric (QP MTU choice, the pool's groups, the switchless relay's ring and credits), the S2 batch planner
# (engine/runtime/s2_batch.h: DFlash2 sessions and their reuse, the insert rule, the MTP confidence stop, the parallel
# drafter step), the scheduler, the KV pool over the Rust page allocator in engine/kvman-rs and its block stores (memory,
# a file, loopback KV nodes), the control protocol, the pack manifest parser, host memory reclaim, and the Flash ring.
# .github/ci/cpu-tests.sh engine (./gr8 test engine) runs it with the other host tests.
#   scripts/host_tests.sh        (JOBS=N sets parallel compile/run concurrency; default: CPU count)
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
TL="$T/logs"; mkdir -p "$TL"
J=${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Build the three Rust static libraries once up front (separate target dirs, so they compile in parallel).
HL=$REPO/engine/target/release/libg6host.a
KN=$REPO/engine/knobs-rs/target/release/libgr8knobs.a
GL=$REPO/engine/loader-rs/target/release/libg6load.a
if command -v cargo > /dev/null 2>&1; then
  rustc -V > /dev/null; cargo -V > /dev/null
  cargo build --release --quiet --manifest-path "$REPO/engine/Cargo.toml" & P1=$!
  cargo build --release --quiet --features capi --manifest-path "$REPO/engine/knobs-rs/Cargo.toml" & P2=$!
  cargo build --release --quiet --manifest-path "$REPO/engine/loader-rs/Cargo.toml" & P3=$!
  wait "$P1"; wait "$P2"; wait "$P3"
fi
[ -f "$HL" ] || { echo "no $HL: cargo build --release --manifest-path engine/Cargo.toml"; exit 1; }
[ -f "$KN" ] || { echo "no $KN: cargo build --release --features capi --manifest-path engine/knobs-rs/Cargo.toml"; exit 1; }
[ -f "$GL" ] || { echo "no $GL: cargo build --release --manifest-path engine/loader-rs/Cargo.toml"; exit 1; }
KVLIB=("$HL" "$KN" -lpthread -ldl -lm)
KNOBS=("$KN" "$HL" -lpthread -ldl -lm)

FMA=""; [ "$(uname -m)" = x86_64 ] && FMA="-mfma"
HAVE_TORCH=0
python3 -c "import torch, numpy" 2>/dev/null && HAVE_TORCH=1 || true

ORDER=()
run_job() {   # run_job <name> <shell-body>: up to J jobs at once, each in its own work dir $D
  local name=$1 body=$2
  ORDER+=("$name")
  while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n 2>/dev/null || sleep 0.1; done
  (
    set -eu
    D="$T/w/$name"
    mkdir -p "$D"
    eval "$body"
  ) > "$TL/$name.log" 2>&1 || touch "$TL/$name.failed" &
}

# ---- Longest tests first (LPT scheduling) ------------------------------------------------------------------------

# engine/scheduler-rs against transcripts recorded from the C++ it replaced, in each model's shape.
# Default shape under ASan+UBSan (-O2 keeps full sanitizer checks while cutting runtime ~2x; sections run in parallel).
run_job scheduler_rs_vectors_asan '
  g++ -std=c++17 -O2 -g -fsanitize=address,undefined -Wall -Wextra -pthread -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/scheduler_rs_vectors_test.cpp" -o "$D/scheduler_rs_vectors_test" "${KVLIB[@]}"
  "$D/scheduler_rs_vectors_test" 2> >(grep -v "^# s2 GR8_CP_STEP_ROWS=" >&2)
'
run_job scheduler_rs_vectors_flash '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -DG6_MODEL_FLASH=1 -DDEC_S2 -DG6_NR=4 -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/scheduler_rs_vectors_test.cpp" -o "$D/scheduler_rs_vectors_test" "${KVLIB[@]}"
  "$D/scheduler_rs_vectors_test" 2> >(grep -v "^# s2 GR8_CP_STEP_ROWS=" >&2)
'
run_job scheduler_rs_vectors_ds41 '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -DG6_MODEL_DS41=1 -DDEC_S2 -DG6_NR=4 -DG6_MX=48 -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/scheduler_rs_vectors_test.cpp" -o "$D/scheduler_rs_vectors_test" "${KVLIB[@]}"
  "$D/scheduler_rs_vectors_test" 2> >(grep -v "^# s2 GR8_CP_STEP_ROWS=" >&2)
'

# module_calls_test and admit_hold_drain_test under ASan+UBSan
run_job module_calls_asan '
  g++ -std=c++17 -O1 -g -Wall -Wextra -pthread -ffp-contract=off -fsanitize=address,undefined -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/module_calls_test.cpp" -o "$D/module_calls_test_asan" "${KVLIB[@]}"
  "$D/module_calls_test_asan" 2>/dev/null
'
run_job admit_hold_drain_asan '
  g++ -std=c++17 -O1 -g -Wall -Wextra -pthread -fsanitize=address,undefined -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/admit_hold_drain_test.cpp" "${KVLIB[@]}" -o "$D/admit_hold_drain_test_asan"
  (cd "$D" && "$D/admit_hold_drain_test_asan" 2>/dev/null)
'

# Inkling-2-Small & Inkling-2-Mini CPU reference & mini checkpoint
run_job test_ink2_cpu_ref '
  g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off "$REPO/engine/tests/test_ink2_cpu_ref.cpp" -o "$D/test_ink2_cpu_ref"
  "$D/test_ink2_cpu_ref"
'
run_job test_ink2_loop_ref '
  g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off "$REPO/engine/tests/test_ink2_loop_ref.cpp" -o "$D/test_ink2_loop_ref"
  "$D/test_ink2_loop_ref"
'
run_job test_ink2_mini_ckpt '
  g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off "$REPO/engine/tests/test_ink2_mini_ckpt.cpp" \
    "$REPO/engine/loader/g6pack.cpp" "$GL" -lpthread -ldl -lm -o "$D/test_ink2_mini_ckpt"
  "$D/test_ink2_mini_ckpt"
'

# Pool stash / namespace / kv_share tests over loopback kvman-remote nodes
run_job pool_stores_and_share '
  NODE_PID=""; LAZY_PID=""; SHARE_PID=""
  trap '\''[ -n "$NODE_PID" ] && kill "$NODE_PID" 2>/dev/null || true; [ -n "$LAZY_PID" ] && kill "$LAZY_PID" 2>/dev/null || true; [ -n "$SHARE_PID" ] && kill "$SHARE_PID" 2>/dev/null || true'\'' EXIT
  g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_stash_test.cpp" "${KVLIB[@]}" -o "$D/pool_stash_test"
  STORES=("mem:" "ssd:$D/store")
  NODE=$REPO/engine/target/release/kvman-remote
  if [ -x "$NODE" ]; then
    "$NODE" --gb 0.05 --port 0 --bind 127.0.0.1 --min-free-gb 0 --no-lock --ready "$D/node.json" > "$D/node.log" 2>&1 &
    NODE_PID=$!
    for _ in $(seq 50); do [ -s "$D/node.json" ] && break; sleep 0.1; done
    PORT=$(sed -n "s/.*\"port\":\([0-9]*\).*/\1/p" "$D/node.json")
    [ -n "$PORT" ] && STORES+=("remote:127.0.0.1:$PORT")
    LOCK=--no-lock; [ "$(ulimit -l)" = unlimited ] || [ "$(ulimit -l)" -ge 131072 ] && LOCK=
    "$NODE" --gb 0.05 --port 0 --bind 127.0.0.1 --min-free-gb 0 $LOCK --lazy-lock --ready "$D/lazy.json" > "$D/lazy.log" 2>&1 &
    LAZY_PID=$!
    for _ in $(seq 50); do [ -s "$D/lazy.json" ] && break; sleep 0.1; done
    LPORT=$(sed -n "s/.*\"port\":\([0-9]*\).*/\1/p" "$D/lazy.json")
    [ -n "$LPORT" ] && STORES+=("remote:127.0.0.1:$LPORT")
  fi
  "$D/pool_stash_test" "${STORES[@]}"
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_namespace_test.cpp" "${KVLIB[@]}" -o "$D/pool_namespace_test"
  "$D/pool_namespace_test" "${STORES[@]}"
  [ -n "$NODE_PID" ] && kill "$NODE_PID" && NODE_PID=""
  [ -n "$LAZY_PID" ] && kill "$LAZY_PID" && LAZY_PID=""
  if [ -x "$NODE" ]; then
    g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/kv_share_test.cpp" "${KVLIB[@]}" -o "$D/kv_share_test"
    "$NODE" --gb 0.05 --seg-mb 2 --margin-s 0 --port 0 --bind 127.0.0.1 --min-free-gb 0 --no-lock --no-rdma --ready "$D/share.json" > "$D/share.log" 2>&1 &
    SHARE_PID=$!
    for _ in $(seq 50); do [ -s "$D/share.json" ] && break; sleep 0.1; done
    SPORT=$(sed -n "s/.*\"port\":\([0-9]*\).*/\1/p" "$D/share.json")
    "$D/kv_share_test" "remote:127.0.0.1:$SPORT"
    kill "$SHARE_PID" && SHARE_PID=""
  fi
'

# ---- Remaining host tests (each independent compile + run) -------------------------------------------------------

run_job nvcc_cache_test 'bash "$REPO/engine/runtime/tests/nvcc_cache_test.sh"'
run_job ds41_build_flags_test 'bash "$REPO/engine/runtime/tests/ds41_build_flags_test.sh"'

for test in test_ds41_v2_l20_tail test_ds41_v2_send_ffn_opt test_ds41_v2_bf16_tc test_ds41_v2_width test_ds41_v2_attn_pipe; do
  run_job "$test" "
    g++ -std=c++17 -O2 -Wall -Wextra \"\$REPO/engine/tests/$test.cpp\" -o \"\$D/$test\"
    \"\$D/$test\"
  "
done

run_job steering_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/steering_test.cpp" \
    "$REPO/engine/loader/g6pack.cpp" -o "$D/steering_test"
  "$D/steering_test" "$REPO/engine/runtime/tests/fixtures/steering_contract.json"
'
run_job steering_glp_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/steering_glp_test.cpp" \
    "$REPO/engine/loader/g6pack.cpp" -o "$D/steering_glp_test"
  "$D/steering_glp_test"
'
run_job fabric_mtu_test '
  g++ -std=c++17 -O2 -Wall -I "$REPO/engine/comms" "$REPO/engine/comms/fabric_mtu_test.cpp" -o "$D/fabric_mtu_test"
  "$D/fabric_mtu_test"
'
run_job fabric_order_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/comms" "$REPO/engine/comms/tests/fabric_order_test.cpp" -o "$D/fabric_order_test"
  "$D/fabric_order_test"
'
run_job fabric_balance_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/comms" "$REPO/engine/comms/tests/fabric_balance_test.cpp" -o "$D/fabric_balance_test"
  "$D/fabric_balance_test"
'
run_job fabric_runs_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/comms" "$REPO/engine/comms/tests/fabric_runs_test.cpp" -o "$D/fabric_runs_test"
  "$D/fabric_runs_test"
'
run_job fabric_ring_runs_test '
  g++ -std=c++17 -O2 -Wall -Wextra -fsanitize=address,undefined -I "$REPO/engine/comms" "$REPO/engine/comms/tests/fabric_ring_runs_test.cpp" -o "$D/fabric_ring_runs_test"
  "$D/fabric_ring_runs_test"
'
run_job relay_test '
  g++ -std=c++17 -O2 -Wall -fsanitize=address,undefined -I "$REPO/engine/comms" "$REPO/engine/comms/tests/relay_test.cpp" -o "$D/relay_test"
  "$D/relay_test"
'
run_job relay_loop_test 'python3 "$REPO/engine/comms/tests/relay_loop_test.py"'
run_job fabric_pool_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/comms" "$REPO/engine/comms/tests/fabric_pool_test.cpp" -o "$D/fabric_pool_test"
  "$D/fabric_pool_test"
'
run_job knobs_header_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/knobs-rs" "$REPO/engine/knobs-rs/tests/header_test.cpp" "${KNOBS[@]}" -o "$D/knobs_header_test"
  "$D/knobs_header_test"
'
run_job test_s6_kv4_host '
  g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off "$REPO/engine/tests/test_s6_kv4_host.cpp" "${KNOBS[@]}" -o "$D/test_s6_kv4_host"
  "$D/test_s6_kv4_host"
'
run_job test_dispram_host '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-sanitize-recover=undefined -pthread \
    -I "$REPO/engine" "$REPO/engine/tests/test_dispram_host.cpp" "${KNOBS[@]}" -o "$D/test_dispram_host"
  "$D/test_dispram_host"
'
run_job grammar_request_test '
  g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/grammar_request_test.cpp" -o "$D/grammar_request_test" "${KNOBS[@]}"
  "$D/grammar_request_test"
'
run_job penalty_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/penalty_test.cpp" -o "$D/penalty_test" "${KNOBS[@]}"
  "$D/penalty_test"
'
run_job image_request_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/image_request_test.cpp" -o "$D/image_request_test" "${KNOBS[@]}"
  "$D/image_request_test"
'
run_job image_state_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/image_state_test.cpp" -o "$D/image_state_test" "${KNOBS[@]}"
  "$D/image_state_test"
'
run_job ds41_engram_ring_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-sanitize-recover=undefined -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/ds41_engram_ring_test.cpp" -o "$D/ds41_engram_ring_test"
  "$D/ds41_engram_ring_test"
'
run_job batch_round_test '
  g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_round_test.cpp" -o "$D/batch_round_test" "${KNOBS[@]}"
  "$D/batch_round_test"
'
run_job batch_k0_test '
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined -D_GLIBCXX_ASSERTIONS -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_k0_test.cpp" -o "$D/batch_k0_test" "${KNOBS[@]}"
  "$D/batch_k0_test" > /dev/null
'
for t in batch_dflash_test batch_dfl_verify_cap_test batch_dsel_test batch_dflreuse_test batch_insert_test batch_kstop_test batch_draftpar_test batch_dselmodel_test batch_kstop_mtp_test scheduler_fair_test scheduler_r_test scheduler_horizon_test; do
  run_job "$t" "
    g++ -std=c++17 -O2 -Wall -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/$t.cpp\" -o \"\$D/$t\" \"\${KNOBS[@]}\"
    \"\$D/$t\"
  "
done
run_job batch_ngram_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_ngram_test.cpp" -o "$D/batch_ngram_test" "${KNOBS[@]}"
  "$D/batch_ngram_test" 2>/dev/null
'
run_job batch_kstopb_test '
  g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_kstopb_test.cpp" -o "$D/batch_kstopb_test" "${KNOBS[@]}"
  "$D/batch_kstopb_test" 2>/dev/null
'
run_job planner_facade_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/planner_facade_test.cpp" -o "$D/planner_facade_test" "${KNOBS[@]}"
  "$D/planner_facade_test"
'
run_job s2_dsel_table 'python3 "$REPO/tools/s2_dsel_table.py" --selftest'

# module_info_test in each model's shape
i=0
for f in "" "-DDEC_S2 -DG6_NR=4" "-DG6_MODEL_FLASH=1 -DDEC_S2 -DG6_NR=4" "-DG6_MODEL_DS41=1 -DDEC_S2 -DG6_NR=4 -DG6_MX=48" \
         "-DG6_MODEL_INK2=1 -DDEC_S2 -DG6_NR=4" "-DG6_MODEL_INK2=1 -DDEC_S2 -DG6_NR=6" "-DG6_MODEL_INK2=1 -DDEC_S2 -DG6_NR=1" \
         "-DG6_MODEL_HY4=1 -DDEC_S2 -DG6_NR=1" "-DG6_MODEL_HY4=1 -DDEC_S2 -DG6_NR=2"; do
  run_job "module_info_$i" "
    g++ -std=c++17 -O2 -Wall -Wextra $f -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/tests/module_info_test.cpp\" -o \"\$D/module_info_test\" \"\${KVLIB[@]}\"
    \"\$D/module_info_test\"
  "
  i=$((i + 1))
done

# batch_replay_test + round_replay off/fast
run_job batch_replay_and_round '
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Wall -Wextra -ffp-contract=off -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_replay_test.cpp" -o "$D/batch_replay_test" "${KNOBS[@]}"
  "$D/batch_replay_test" "$D/replay.log"
  GR8_S2_REPLAY_LOG="$D/env.log" "$D/batch_replay_test"
  for c in off fast; do
    g++ -std=c++17 -O2 -Wall -ffp-contract=$c $([ $c = fast ] && echo $FMA) -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/round_replay.cpp" -o "$D/round_replay_$c" "${KNOBS[@]}"
  done
  "$D/round_replay_off" "$D/replay.log"
  "$D/round_replay_fast" --steps-only "$D/replay.log"
'

run_job ngram_lookup_check '
  g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/ngram_lookup_check.cpp" -o "$D/ngram_lookup_check" "${KNOBS[@]}"
  (cd "$REPO/tools/ngram/tests" && NGRAM_LOOKUP_CHECK="$D/ngram_lookup_check" python3 -m unittest -q test_ngram.ExportTest)
'
run_job flash_dfl_window_test '
  g++ -std=c++17 -O2 -Wall -Wextra -fsanitize=address,undefined -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/flash_dfl_window_test.cpp" \
    -o "$D/flash_dfl_window_test" "${KVLIB[@]}"
  "$D/flash_dfl_window_test"
'
run_job batch_ds41_sessdrafts_test '
  g++ -std=c++17 -O2 -Wall -DG6_MODEL_DS41=1 -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_ds41_sessdrafts_test.cpp" -o "$D/batch_ds41_sessdrafts_test" "${KNOBS[@]}"
  "$D/batch_ds41_sessdrafts_test"
'
run_job capture_barrier_test '
  g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/capture_barrier_test.cpp" -o "$D/capture_barrier_test" "${KNOBS[@]}"
  "$D/capture_barrier_test"
'
for t in pool_reuse_test pool_conc_replay_test driver_grow_test; do
  run_job "$t" "
    g++ -std=c++17 -O2 -Wall -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/$t.cpp\" \"\${KVLIB[@]}\" -o \"\$D/$t\"
    \"\$D/$t\"
  "
done
run_job driver_facade_test '
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/driver_facade_test.cpp" -o "$D/driver_facade_test" "${KVLIB[@]}"
  "$D/driver_facade_test"
'
for t in scheduler_rounds_test scheduler_tail_first_test; do
  run_job "$t" "
    g++ -std=c++17 -O2 -Wall -Wextra -pthread -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/$t.cpp\" -o \"\$D/$t\" \"\${KNOBS[@]}\"
    \"\$D/$t\"
  "
done
run_job scheduler_park_test '
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Wall -Wextra -DG6_MODEL_DS41=1 -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/scheduler_park_test.cpp" -o "$D/scheduler_park_test" "${KNOBS[@]}"
  "$D/scheduler_park_test"
'
run_job ds41_park_evict_test '
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Wall -Wextra -DG6_MODEL_DS41=1 -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/ds41_park_evict_test.cpp" -o "$D/ds41_park_evict_test" "${KVLIB[@]}"
  "$D/ds41_park_evict_test"
  GR8_DS_CACHE_REPLACE=0 "$D/ds41_park_evict_test"
  GR8_DS_CACHE_REPLACE=0 GR8_DS_CACHE_DEDUP=0 "$D/ds41_park_evict_test"
  GR8_DS_CACHE_DEFER_REPLACE=0 "$D/ds41_park_evict_test"
'
run_job test_ds41_prefix_cache '
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Wall -Wextra -DG6_MODEL_DS41=1 -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/test_ds41_prefix_cache.cpp" -o "$D/test_ds41_prefix_cache" "$HL" "$KN" -lpthread -ldl -lm
  "$D/test_ds41_prefix_cache"
  GR8_DS_CACHE_DEFER_REPLACE=0 "$D/test_ds41_prefix_cache"
  GR8_DS_CACHE_REPLACE=0 "$D/test_ds41_prefix_cache"
  GR8_DS_CACHE_DEDUP=0 "$D/test_ds41_prefix_cache"
'
run_job driver_prefill_quantum_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/driver_prefill_quantum_test.cpp" -o "$D/driver_prefill_quantum_test" "${KNOBS[@]}"
  "$D/driver_prefill_quantum_test" > /dev/null
'
run_job kv_plan_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/kv_plan_test.cpp" -o "$D/kv_plan_test" "${KNOBS[@]}"
  "$D/kv_plan_test"
'
# the page block's sizes from kvman-rs (g6kv_layout_sizes) with each build's rows (attn/kvfmt.h): the blocks the stores
# and the Flash checkpoints have kept, byte for byte
run_job kv_layout_test '
  for f in "" "-DG6_KV4=1" "-DG6_MODEL_FLASH=1" "-DG6_MODEL_FLASH=1 -DG6_FLASH_KV_FP8=1" "-DG6_MODEL_FLASH=1 -DG6_KV4=1" "-DG6_MODEL_DS41=1"; do
    g++ -std=c++17 -O2 -Wall -Wextra $f "$REPO/engine/runtime/s2/tests/kv_layout_test.cpp" -o "$D/kv_layout_test" "${KVLIB[@]}"
    "$D/kv_layout_test" > /dev/null
  done
'
run_job cp_rows_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/cp_rows_test.cpp" -o "$D/cp_rows_test" "${KNOBS[@]}"
  "$D/cp_rows_test" 2> >(grep -v "^# s2 GR8_CP_STEP_ROWS=" >&2)
'
run_job kv_state_desc_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/kv_state_desc_test.cpp" "${KVLIB[@]}" -o "$D/kv_state_desc_test"
  "$D/kv_state_desc_test"
'
run_job ink2_kv_state_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/ink2_kv_state_test.cpp" "${KVLIB[@]}" -o "$D/ink2_kv_state_test"
  "$D/ink2_kv_state_test"
'
run_job control_hold_test '
  g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime/s2" "$REPO/engine/runtime/s2/tests/control_hold_test.cpp" -o "$D/control_hold_test" "${KNOBS[@]}"
  "$D/control_hold_test"
'
run_job control_version_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/control_version_test.cpp" -o "$D/control_version_test" "${KNOBS[@]}"
  "$D/control_version_test"
'

# control_vectors_test, service_vectors_test, admit_vectors_test, module_calls_test in each model's shape
i=0
for f in "" "-DG6_MODEL_FLASH=1 -DDEC_S2 -DG6_NR=4" "-DG6_MODEL_DS41=1 -DDEC_S2 -DG6_NR=4 -DG6_MX=48"; do
  run_job "control_vectors_$i" "
    g++ -std=c++17 -O2 -Wall -Wextra -pthread $f -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/control_vectors_test.cpp\" -o \"\$D/control_vectors_test\" \"\${KNOBS[@]}\"
    \"\$D/control_vectors_test\"
  "
  run_job "service_vectors_$i" "
    g++ -std=c++17 -O2 -Wall -pthread $f -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/service_vectors_test.cpp\" -o \"\$D/service_vectors_test\" \"\${KVLIB[@]}\"
    \"\$D/service_vectors_test\" 2> >(grep -v '^# s2 trace: cannot write ' >&2)
  "
  run_job "admit_vectors_$i" "
    g++ -std=c++17 -O2 -Wall -Wextra -pthread $f -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/admit_vectors_test.cpp\" -o \"\$D/admit_vectors_test\" \"\${KVLIB[@]}\"
    \"\$D/admit_vectors_test\"
  "
  run_job "module_calls_$i" "
    g++ -std=c++17 -O2 -Wall -Wextra -pthread -ffp-contract=off $f -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/module_calls_test.cpp\" -o \"\$D/module_calls_test\" \"\${KVLIB[@]}\"
    \"\$D/module_calls_test\" 2>/dev/null
  "
  i=$((i + 1))
done

# options_vectors_test in each shape (coverage checked after wait)
i=0
for f in "" "-DDEC_S2 -DG6_NR=4" "-DG6_KV4=1 -DDEC_S2 -DG6_NR=4" "-DG6_MODEL_FLASH=1 -DDEC_S2 -DG6_NR=4" "-DG6_MODEL_DS41=1 -DDEC_S2 -DG6_NR=4 -DG6_MX=48"; do
  run_job "options_vectors_$i" "
    g++ -std=c++17 -O2 -Wall -Wextra $f -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/options_vectors_test.cpp\" -o \"\$D/options_vectors_test\" \"\${KVLIB[@]}\"
    \"\$D/options_vectors_test\" \"\$REPO/engine/runtime/s2/tests/options_cases.txt\" --dump > \"\$T/options_$i.txt\" || { tail -1 \"\$T/options_$i.txt\"; exit 1; }
    tail -1 \"\$T/options_$i.txt\"
  "
  i=$((i + 1))
done

run_job signature_vectors_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/signature_vectors_test.cpp" -o "$D/signature_vectors_test" "${KVLIB[@]}"
  "$D/signature_vectors_test"
'

i=0
for f in "" "-DDEC_S2 -DG6_NR=4" "-DDEC_S2 -DG6_NR=3" "-DG6_MODEL_FLASH=1 -DDEC_S2 -DG6_NR=4" "-DG6_MODEL_FLASH=1 -DDEC_S2 -DG6_NR=3" \
         "-DG6_MODEL_FLASH=1 -DDEC_S2 -DG6_NR=2" "-DG6_MODEL_DS41=1 -DDEC_S2 -DG6_NR=4 -DG6_MX=48"; do
  run_job "fabric_plan_vectors_$i" "
    g++ -std=c++17 -O2 -Wall -Wextra $f -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/fabric_plan_vectors_test.cpp\" -o \"\$D/fabric_plan_vectors_test\" \"\${KVLIB[@]}\"
    \"\$D/fabric_plan_vectors_test\"
  "
  i=$((i + 1))
done
# EL6 (#154): fabric_plan_apply_group over loopback FabricPool groups in GLM, Flash and DS41 shapes
i=0
for f in "-DDEC_S2 -DG6_NR=4" "-DDEC_S2 -DG6_NR=3" "-DG6_MODEL_FLASH=1 -DDEC_S2 -DG6_NR=4" "-DG6_MODEL_FLASH=1 -DDEC_S2 -DG6_NR=2"; do
  run_job "fabric_plan_group_$i" "
    g++ -std=c++17 -O2 -Wall -Wextra -pthread $f -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/fabric_plan_group_test.cpp\" -o \"\$D/fabric_plan_group_test\" \"\${KVLIB[@]}\"
    \"\$D/fabric_plan_group_test\"
  "
  i=$((i + 1))
done

run_job control_wire_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/control_wire_test.cpp" -o "$D/control_wire_test" "${KNOBS[@]}"
  "$D/control_wire_test" 2> >(grep -v "^g6ctl: control recv failed: ret=0 " >&2)
'
run_job ds41_agree_knobs_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/ds41_agree_knobs_test.cpp" -o "$D/ds41_agree_knobs_test" "${KNOBS[@]}"
  "$D/ds41_agree_knobs_test" "$REPO/engine/runtime/g6m4.cu"
'
run_job agree_vision_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/agree_vision_test.cpp" -o "$D/agree_vision_test" "${KNOBS[@]}"
  "$D/agree_vision_test" "$REPO/engine/runtime/g6m4.cu"
'
run_job steering_state_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/steering_state_test.cpp" "${KVLIB[@]}" -o "$D/steering_state_test"
  "$D/steering_state_test"
'
run_job steering_transport_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/steering_transport_test.cpp" -o "$D/steering_transport_test" "${KNOBS[@]}"
  "$D/steering_transport_test"
'
run_job trace_test_merge '(cd "$REPO" && python3 -m unittest -q tools.trace.test_merge)'
run_job manifest_test '
  g++ -std=c++20 -O2 -Wall -pthread -I "$REPO/engine/loader" "$REPO/engine/loader/tests/manifest_test.cpp" \
    "$REPO/engine/loader/g6pack.cpp" -o "$D/manifest_test"
  "$D/manifest_test"
'
run_job test_context_buckets '
  g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/test_context_buckets.cpp" -o "$D/test_context_buckets" "${KNOBS[@]}"
  "$D/test_context_buckets"
'
run_job host_reclaim_test '
  g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/host_reclaim_test.cpp" -o "$D/host_reclaim_test" "${KNOBS[@]}"
  "$D/host_reclaim_test"
'
# EL7 (#155) in-process width-module lifecycle & CaptureAhead vs CaptureAfter graph capture policy
run_job width_module_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined "$REPO/engine/runtime/tests/width_module_test.cpp" -o "$D/width_module_test"
  "$D/width_module_test"
'
run_job test_flash_ring '
  g++ -std=c++17 -O2 -Wall "$REPO/engine/tests/test_flash_ring.cpp" -o "$D/test_flash_ring" "${KNOBS[@]}"
  "$D/test_flash_ring"
'
run_job flash_ckpt_index_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/flash_ckpt_index_test.cpp" -o "$D/flash_ckpt_index_test"
  TMPDIR="$D" "$D/flash_ckpt_index_test"
'
run_job flash_ckpt_agree_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -pthread -fsanitize=address,undefined -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/flash_ckpt_agree_test.cpp" -o "$D/flash_ckpt_agree_test" "${KNOBS[@]}"
  TMPDIR="$D" "$D/flash_ckpt_agree_test"
'
run_job flash_rewidth_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/flash_rewidth_test.cpp" -o "$D/flash_rewidth_test" "${KVLIB[@]}"
  TMPDIR="$D" "$D/flash_rewidth_test" "$REPO/engine/target/release/kvman-peer"
'
# EL4d (#161): canonical head-split, DFlash2 ring, KDA linear-attention + conv state (vs kda::heads) and session envelope C FFI
run_job canon_state_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/canon_state_test.cpp" -o "$D/canon_state_test" "${KVLIB[@]}"
  "$D/canon_state_test"
'
run_job kda_carry_cpu_test '
  g++ -std=c++17 -O2 -ffp-contract=off -DG6_MODEL_FLASH=1 "$REPO/engine/kda/tests/kda_carry_cpu_test.cpp" -o "$D/kda_carry_cpu_test" "${KNOBS[@]}"
  "$D/kda_carry_cpu_test"
'
for t in control_tp2_test control_tp1_test flash_lend_test; do
  run_job "$t" "
    g++ -std=c++17 -O2 -Wall -pthread -I \"\$REPO/engine/runtime/s2\" \"\$REPO/engine/runtime/s2/tests/$t.cpp\" -o \"\$D/$t\" \"\${KNOBS[@]}\"
    \"\$D/$t\"
  "
done
run_job admit_hold_drain_test '
  g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/admit_hold_drain_test.cpp" "${KVLIB[@]}" -o "$D/admit_hold_drain_test"
  (cd "$D" && "$D/admit_hold_drain_test" 2>/dev/null)
'
# live session KV export & cross-width import (#159): compact owned-page plans, multi-rank CP/replicated transforms,
# mid-decode Service EXPORT, cross-width IMPORT with zero prefill and exact continuation, and intake refusals
run_job session_export_import_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -pthread -fsanitize=address,undefined -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/session_export_import_test.cpp" "${KVLIB[@]}" -o "$D/session_export_import_test"
  "$D/session_export_import_test" 2>/dev/null
'
# EL4c (#160) two-phase live bulk-then-tail session migration (8 concurrent sessions, 4->3, 3->4, 4->6, mid-bulk abort)
run_job live_migrate_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -pthread -fsanitize=address,undefined -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/live_migrate_test.cpp" "${KVLIB[@]}" -o "$D/live_migrate_test"
  "$D/live_migrate_test"
'
# EL8 (#156) in-process live width switch coordinator (pre-capture, deferred capture, rollbacks) & multi-rank Service SWITCH fence
run_job live_switch_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -pthread -fsanitize=address,undefined -I "$REPO/engine/runtime" \
    "$REPO/engine/runtime/s2/tests/live_switch_test.cpp" "${KVLIB[@]}" -o "$D/live_switch_test"
  "$D/live_switch_test" 2>/dev/null
'
run_job ds41_geom_dump '
  g++ -std=c++17 -O2 -Wall -DG6_MODEL_DS41=1 -I "$REPO/engine" "$REPO/engine/tests/ds41_geom_dump.cpp" -o "$D/ds41_geom_dump"
  (cd "$REPO/engine/tests" && DS41_GEOM_DUMP="$D/ds41_geom_dump" python3 -m unittest -q test_ds41_geom)
'
run_job test_ds41_comp_commit '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine" "$REPO/engine/tests/test_ds41_comp_commit.cpp" -o "$D/test_ds41_comp_commit"
  "$D/test_ds41_comp_commit"
'
if [ -f /usr/local/cuda/include/cuda_runtime.h ]; then
  run_job ds41_attn_bind_test '
    g++ -std=c++17 -O1 -I /usr/local/cuda/include -I "$REPO/engine" "$REPO/engine/attn/tests/ds41_attn_bind_test.cpp" -o "$D/ds41_attn_bind_test"
    "$D/ds41_attn_bind_test" "$REPO"/engine/attn/tests/data/attn_plan_tp*.txt
  '
else
  echo "SKIP ds41_attn_bind_test: no CUDA headers"
fi
run_job ds41_ring_keep_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/attn" "$REPO/engine/attn/tests/ds41_ring_keep_test.cpp" -o "$D/ds41_ring_keep_test"
  "$D/ds41_ring_keep_test"
'
if [ "$HAVE_TORCH" = 1 ]; then
  run_job ds41_torch_refs '
    (cd "$REPO/engine/tests" && python3 -m unittest -q test_ds41_idx_ref)
    python3 "$REPO/engine/ref/ds41/ds41_kernels_selftest.py"
  '
  run_job ds_mhc_cpu_test 'bash "$REPO/engine/runtime/tests/ds_mhc_cpu_test.sh" "$D/ds_mhc_cpu_test"'
else
  echo "SKIP test_ds41_idx_ref, ds41_kernels_selftest, ds_mhc_cpu_test: no torch/numpy"
fi
run_job engram_ring_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/engram" "$REPO/engine/engram/tests/test_ring.cpp" -o "$D/engram_ring_test"
  "$D/engram_ring_test"
'
run_job engram_wide_test '
  g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -pthread -I "$REPO/engine/engram" "$REPO/engine/engram/tests/test_engram_wide.cpp" -o "$D/engram_wide_test"
  "$D/engram_wide_test" "$D/engram_wide"
'
run_job engram_rowstore_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/engram" "$REPO/engine/engram/tests/test_rowstore.cpp" -o "$D/engram_rowstore_test"
  "$D/engram_rowstore_test" "$D/engram_rowstore" 262144
'
run_job qw38_ple_trellis_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/engram" "$REPO/engine/engram/tests/test_qw38_ple_trellis.cpp" -o "$D/qw38_ple_trellis_test"
  "$D/qw38_ple_trellis_test" "$D/qw38_ple_trellis"
'
run_job engram_uring_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread "$REPO/engine/engram/tests/test_uring.cpp" -o "$D/engram_uring_test"
  "$D/engram_uring_test"
'
run_job engram_boot_test '
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine" "$REPO/engine/engram/tests/test_boot.cpp" -o "$D/engram_boot_test" "${KNOBS[@]}"
  "$D/engram_boot_test"
'
run_job engram_col_shard_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine" "$REPO/engine/engram/tests/test_col_shard.cpp" -o "$D/engram_col_shard_test" "${KNOBS[@]}"
  mkdir -p "$D/engram_col_shard"
  "$D/engram_col_shard_test" "$D/engram_col_shard"
'
run_job engram_sparse_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine" "$REPO/engine/engram/tests/test_sparse.cpp" -o "$D/engram_sparse_test" "${KNOBS[@]}"
  "$D/engram_sparse_test" "$D/engram_sparse"
'
if [ -f /usr/local/cuda/include/cuda_fp8.h ]; then
  run_job engram_act_quant_test '
    g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -I "$REPO/engine/engram" -I /usr/local/cuda/include \
      "$REPO/engine/engram/tests/test_act_quant.cpp" -o "$D/engram_act_quant_test"
    "$D/engram_act_quant_test"
  '
fi
run_job ds_multitile_host_test '
  g++ -std=c++17 -O2 -Wall -ffp-contract=off "$REPO/engine/kernels/tests/ds_multitile_host_test.cpp" -o "$D/ds_multitile_host_test"
  "$D/ds_multitile_host_test"
'
run_job ds_ro_host_test '
  g++ -std=c++17 -O2 -Wall "$REPO/engine/kernels/tests/ds_ro_host_test.cpp" -o "$D/ds_ro_host_test"
  "$D/ds_ro_host_test"
'
run_job test_ds41_w4096 '
  g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -I "$REPO/engine" -I "$REPO/engine/kernels" "$REPO/engine/tests/test_ds41_w4096.cpp" -o "$D/test_ds41_w4096"
  "$D/test_ds41_w4096"
'
for name in test_ds41_v2 test_ds41_v2_walk test_ds41_v2_plan; do
  run_job "$name" "
    g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -DG6_NR=4 -DDEC_S2 -I \"\$REPO/engine\" \"\$REPO/engine/tests/$name.cpp\" -o \"\$D/$name\" \"\${KVLIB[@]}\"
    \"\$D/$name\"
  "
done
run_job test_ds41_pf_split '
  g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -I "$REPO/engine" "$REPO/engine/tests/test_ds41_pf_split.cpp" -o "$D/test_ds41_pf_split"
  "$D/test_ds41_pf_split" > "$D/pf_split.log" || { cat "$D/pf_split.log"; exit 1; }
  tail -1 "$D/pf_split.log"
'
run_job test_ds41_lanes '
  g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -I "$REPO/engine" -I "$REPO/engine/runtime" -I "$REPO/engine/comms" -I "$REPO/engine/kernels" \
    "$REPO/engine/tests/test_ds41_lanes.cpp" -o "$D/test_ds41_lanes" "${KNOBS[@]}"
  "$D/test_ds41_lanes" > "$D/lanes.log" || { cat "$D/lanes.log"; exit 1; }
  tail -1 "$D/lanes.log"
  "$D/test_ds41_lanes" l20 > "$D/lanes_l20.log" || { cat "$D/lanes_l20.log"; exit 1; }
  tail -1 "$D/lanes_l20.log"
'
run_job test_flash_serve_py '
  python3 "$REPO/engine/tests/test_flash_serve_check.py"
  python3 "$REPO/engine/tests/test_flash_serve_prefill_check.py"
  python3 "$REPO/engine/tests/test_flash_serve_waterfill.py"
'
run_job pf_peer_stage_test '
  g++ -std=c++17 -O2 -Wall "$REPO/engine/runtime/tests/pf_peer_stage_test.cpp" -o "$D/pf_peer_stage_test" "${KNOBS[@]}"
  "$D/pf_peer_stage_test"
'
run_job test_topk '
  g++ -std=c++17 -O2 -I "$REPO/engine/kernels" "$REPO/engine/tests/test_topk.cpp" -o "$D/test_topk" "${KNOBS[@]}"
  "$D/test_topk"
'
run_job test_moe_union_cap '
  g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off "$REPO/engine/tests/test_moe_union_cap.cpp" -o "$D/test_moe_union_cap"
  "$D/test_moe_union_cap" > "$D/test_moe_union_cap.out" || { cat "$D/test_moe_union_cap.out"; exit 1; }
  head -n 2 "$D/test_moe_union_cap.out"
'
run_job ds41_prefill_test '
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -DG6_MODEL_DS41=1 -I "$REPO/engine/runtime" -I "$REPO/engine/comms" "$REPO/engine/runtime/s2/tests/ds41_prefill_test.cpp" -o "$D/ds41_prefill_test" "${KNOBS[@]}"
  "$D/ds41_prefill_test"
'
for t in ds41_park_miss_test ds41_prefill_lanes_test ds41_s32_step_test s32_64k_mtp8_test batch_ds41_dspark_test ds41_reuse_mtp0_test; do
  run_job "$t" "
    g++ -std=c++17 -O2 -Wall -DG6_MODEL_DS41=1 -I \"\$REPO/engine/runtime\" -I \"\$REPO/engine/runtime/s2\" -I \"\$REPO/engine\" \"\$REPO/engine/runtime/s2/tests/$t.cpp\" -o \"\$D/$t\" \"\${KVLIB[@]}\"
    \"\$D/$t\" > \"\$D/$t.out\" 2>&1 || { cat \"\$D/$t.out\"; exit 1; }
  "
done
for t in batch_dflash_test batch_dflreuse_test batch_draftpar_test batch_dsel_test batch_dselmodel_test batch_insert_test \
         batch_kstop_test batch_kstopb_test batch_kstop_mtp_test batch_ngram_test batch_round_test batch_replay_test \
         driver_grow_test driver_prefill_quantum_test scheduler_fair_test scheduler_rounds_test scheduler_r_test \
         scheduler_tail_first_test cp_rows_test grammar_request_test kv_plan_test; do
  run_job "flash_$t" "
    g++ -std=c++17 -O1 -Wall -pthread -DG6_MODEL_FLASH=1 -I \"\$REPO/engine/runtime\" \"\$REPO/engine/runtime/s2/tests/$t.cpp\" -o \"\$D/flash_$t\" \"\${KVLIB[@]}\"
    (cd \"\$D\" && \"\$D/flash_$t\" > \"\$D/flash_$t.out\" 2>&1) || { cat \"\$D/flash_$t.out\"; echo \"flash build: $t failed\"; exit 1; }
  "
done
for t in control_sync_test control_idle_test; do
  run_job "$t" "
    g++ -std=c++17 -O2 -pthread -I \"\$REPO/engine/runtime/s2\" \"\$REPO/engine/runtime/s2/tests/$t.cpp\" -o \"\$D/$t\" \"\${KNOBS[@]}\"
    \"\$D/$t\"
  "
done
run_job test_progress_cpus '
  g++ -std=c++20 -O1 -pthread -I "$REPO/engine/comms" "$REPO/tools/health/test_progress_cpus.cpp" -o "$D/test_progress_cpus" "${KNOBS[@]}"
  "$D/test_progress_cpus"
'
run_job cp_batch_cpu_test '
  g++ -std=c++17 -O2 "$REPO/engine/runtime/tests/cp_batch_cpu_test.cpp" -o "$D/cp_batch_cpu_test" "${KNOBS[@]}"
  "$D/cp_batch_cpu_test"
'
run_job flash_dsa_full_schedule_test '
  g++ -std=c++17 -O2 "$REPO/engine/runtime/tests/flash_dsa_full_schedule_test.cpp" -o "$D/flash_dsa_full_schedule_test" "${KNOBS[@]}"
  "$D/flash_dsa_full_schedule_test"
'
run_job test_flash_pool_owner '
  g++ -std=c++17 -O2 "$REPO/engine/tests/test_flash_pool_owner.cpp" -o "$D/test_flash_pool_owner" "${KNOBS[@]}"
  "$D/test_flash_pool_owner"
'
run_job ink2_moe_host_test '
  g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off "$REPO/engine/kernels/tests/ink2_moe_host_test.cpp" -o "$D/ink2_moe_host_test"
  "$D/ink2_moe_host_test"
'
run_job test_ink2_geom '
  g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_INK2=1 -I "$REPO/engine" "$REPO/engine/tests/test_ink2_geom.cpp" -o "$D/test_ink2_geom" "${KNOBS[@]}"
  "$D/test_ink2_geom"
'
run_job test_ink2_dflash2_melded '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -DG6_MODEL_INK2=1 -I "$REPO/engine" -I "$REPO/engine/runtime" \
    "$REPO/engine/tests/test_ink2_dflash2_melded.cpp" -o "$D/test_ink2_dflash2_melded" "${KVLIB[@]}"
  "$D/test_ink2_dflash2_melded"
'
# EL5 (#153) resizable KV pool chunk coordination (kv_chunks_plan.h) against kvman-rs RunPool
run_job kv_chunks_plan_test '
  g++ -std=c++17 -O2 -Wall -Wextra -fsanitize=address,undefined -I "$REPO/engine/runtime" -I "$REPO/engine/runtime/s2" -I "$REPO/engine" \
    "$REPO/engine/runtime/s2/tests/kv_chunks_plan_test.cpp" -o "$D/kv_chunks_plan_test" "${KVLIB[@]}"
  "$D/kv_chunks_plan_test"
'
# EL3 (#151) piece-table slab pool (chunk release, contiguous take_run, direct-load RouteMap)
run_job ptab_pool_test '
  g++ -std=c++17 -O2 -Wall -Wextra -fsanitize=address,undefined "$REPO/engine/kernels/tests/ptab_pool_test.cpp" -o "$D/ptab_pool_test"
  "$D/ptab_pool_test"
'
# EL3 (#151 Stage 3) delta piece-prefetch planner + g6l_delta C FFI (4->3, 3->4, joiner, mid-prefetch rollback)
run_job ptab_delta_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined "$REPO/engine/kernels/tests/ptab_delta_test.cpp" "$GL" \
    -lpthread -ldl -lm -o "$D/ptab_delta_test"
  "$D/ptab_delta_test"
'
# EL3b (#164) online per-step weight defragmentation: fenced retirement, shared-expert refresh, KV headroom borrow
run_job ptab_compact_test '
  g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined "$REPO/engine/kernels/tests/ptab_compact_test.cpp" -o "$D/ptab_compact_test"
  "$D/ptab_compact_test"
'
# Native dense embedding engine (g6embed): protocol framing, M-RoPE geometry, base64 RGB decode, token-budget batching
run_job embed_proto_test '
  g++ -std=c++17 -O2 -Wall -Wextra -fsanitize=address,undefined -pthread -I "$REPO/engine/embed" \
    "$REPO/engine/embed/tests/embed_proto_test.cpp" -o "$D/embed_proto_test"
  "$D/embed_proto_test"
'

# Tencent Hy4-flash geometry, math, Engram hash/PCG64, and DFly2 spec-decode host unit tests
run_job hy4_unit_tests '
  g++ -std=c++20 -O2 -Wall -Wextra -DG6_MODEL_HY4=1 -I "$REPO/engine" "$REPO/engine/tests/test_hy4_geom.cpp" -o "$D/test_hy4_geom" && "$D/test_hy4_geom"
  g++ -std=c++20 -O2 -Wall -Wextra -I "$REPO/engine" "$REPO/engine/tests/test_hy4_math.cpp" -o "$D/test_hy4_math" && "$D/test_hy4_math"
  g++ -std=c++20 -O2 -Wall -Wextra -I "$REPO/engine" "$REPO/engine/tests/test_hy4_engram.cpp" -o "$D/test_hy4_engram" && "$D/test_hy4_engram"
  g++ -std=c++20 -O2 -Wall -Wextra -I "$REPO/engine" "$REPO/engine/tests/test_hy4_dfly.cpp" -o "$D/test_hy4_dfly" && "$D/test_hy4_dfly"
'

wait
for name in "${ORDER[@]}"; do
  [ -s "$TL/$name.log" ] && cat "$TL/$name.log"
done
if ls "$TL"/*.failed > /dev/null 2>&1; then
  echo "host_tests.sh failed: $(cd "$TL" && ls -- *.failed | sed 's/\.failed$//' | tr '\n' ' ')" >&2
  exit 1
fi

python3 "$REPO/scripts/check_options_coverage.py" "$REPO/engine/scheduler-rs/src/options.rs" "$T"/options_*.txt

# the bare environment reads the knob table replaced, and the table against the tree
# CI sets HOST_TESTS_SKIP_ENV_LINT=1 (non-GR8_ prefix so engine/knobs-rs does not reject it as an unknown engine knob)
# because the dedicated env-lint job runs tools/env_lint.py on every push/PR.
if [ "${HOST_TESTS_SKIP_ENV_LINT:-0}" != 1 ]; then
  python3 "$REPO/tools/env_lint.py"
fi
