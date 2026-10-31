#!/bin/bash
# Host-only tests (g++ and cargo, no CUDA) of the engine's CPU-side logic: the build's object cache, steering contracts,
# the fabric (QP MTU choice, the pool's groups, the switchless relay's ring and credits), the S2 batch planner
# (engine/runtime/s2_batch.h: DFlash2 sessions and their reuse, the insert rule, the MTP confidence stop, the parallel
# drafter step), the scheduler, the KV pool over the Rust page allocator in engine/kvman-rs and its block stores (memory,
# a file, loopback KV nodes), the control protocol, the pack manifest parser, host memory reclaim, and the Flash ring,
# checkpoint and prefill split rules.
# .github/ci/cpu-tests.sh engine (./gr8 test engine) runs it with the other host tests.
#   scripts/host_tests.sh
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
# the build's object cache (engine/runtime/nvcc_cache.sh, #32), with g++ standing in for nvcc
bash "$REPO/engine/runtime/tests/nvcc_cache_test.sh"
# Inactive steering contracts/profile loader: CPU only, no CUDA or Rust dependencies.
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/steering_test.cpp" \
  "$REPO/engine/loader/g6pack.cpp" -o "$T/steering_test"
"$T/steering_test" "$REPO/engine/runtime/tests/fixtures/steering_contract.json"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/comms" "$REPO/engine/comms/fabric_mtu_test.cpp" -o "$T/fabric_mtu_test"
"$T/fabric_mtu_test"
# the proxy's collective order (engine/comms/fabric_order.h): a plan that publishes out of numeric order stalls without
# Rs::order and passes with it; no order is the numeric loop
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/comms" "$REPO/engine/comms/tests/fabric_order_test.cpp" -o "$T/fabric_order_test"
"$T/fabric_order_test"
# slot runs and the compact recv area (fabric.h fab_slot_layout): one run is the old offsets, runs tile the region
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/comms" "$REPO/engine/comms/tests/fabric_runs_test.cpp" -o "$T/fabric_runs_test"
"$T/fabric_runs_test"
g++ -std=c++17 -O2 -Wall -Wextra -fsanitize=address,undefined -I "$REPO/engine/comms" "$REPO/engine/comms/tests/fabric_ring_runs_test.cpp" -o "$T/fabric_ring_runs_test"
"$T/fabric_ring_runs_test"
g++ -std=c++17 -O2 -Wall -fsanitize=address,undefined -I "$REPO/engine/comms" "$REPO/engine/comms/tests/relay_test.cpp" \
  -o "$T/relay_test"
"$T/relay_test"
python3 "$REPO/engine/comms/tests/relay_loop_test.py"
# the fabric pool (engine/comms/fabric_pool.h, #154): boot, groups and their ports over loopback, no leaked descriptor
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/comms" "$REPO/engine/comms/tests/fabric_pool_test.cpp" -o "$T/fabric_pool_test"
"$T/fabric_pool_test"
# the Rust page allocator (engine/kvman-rs) that s2/pool.h wraps, built here where cargo is on the PATH
KV=$REPO/engine/kvman-rs/target/release/libg6kvman.a
# and the knob table's C API (engine/knobs-rs), which every test that reads a knob links
KN=$REPO/engine/knobs-rs/target/release/libgr8knobs.a
if command -v cargo > /dev/null 2>&1; then
  cargo build --release --quiet --manifest-path "$REPO/engine/kvman-rs/Cargo.toml"
  cargo build --release --quiet --features capi --manifest-path "$REPO/engine/knobs-rs/Cargo.toml"
fi
[ -f "$KV" ] || { echo "no $KV: cargo build --release --manifest-path engine/kvman-rs/Cargo.toml"; exit 1; }
[ -f "$KN" ] || { echo "no $KN: cargo build --release --features capi --manifest-path engine/knobs-rs/Cargo.toml"; exit 1; }
KVLIB=("$KV" "$KN" -lpthread -ldl -lm)
KNOBS=("$KN" -lpthread -ldl -lm)
# the knob table: its C++ header over the C API
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/knobs-rs" "$REPO/engine/knobs-rs/tests/header_test.cpp" "${KNOBS[@]}" \
  -o "$T/knobs_header_test"
"$T/knobs_header_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/grammar_request_test.cpp" -o "$T/grammar_request_test" "${KNOBS[@]}"
"$T/grammar_request_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_round_test.cpp" -o "$T/batch_round_test" "${KNOBS[@]}"
"$T/batch_round_test"
# K = 0 (no draft head): a Batch runs no DRAFT or CHAIN step and never writes past St::d, and a Driver serves requests
# at mtp 0 with every prompt as decode steps; the K >= 1 digests stay what they were
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -D_GLIBCXX_ASSERTIONS -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_k0_test.cpp" -o "$T/batch_k0_test" "${KNOBS[@]}"
"$T/batch_k0_test" > /dev/null
# a packed prompt tail: one MAIN step for several sessions' tails, each emitting and inserting on its own
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_c8_tail_test.cpp" -o "$T/batch_c8_tail_test" "${KNOBS[@]}"
"$T/batch_c8_tail_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_dflash_test.cpp" -o "$T/batch_dflash_test" "${KNOBS[@]}"
"$T/batch_dflash_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_dsel_test.cpp" -o "$T/batch_dsel_test" "${KNOBS[@]}"
"$T/batch_dsel_test"
# the n-gram table as a draft source (GR8_NGRAM, #56): the table file, chains, the round choice, exact output
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_ngram_test.cpp" -o "$T/batch_ngram_test" "${KNOBS[@]}"
"$T/batch_ngram_test" 2>/dev/null    # the GR8_MTP_LPLOG case prints its round lines
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/ngram_lookup_check.cpp" -o "$T/ngram_lookup_check" "${KNOBS[@]}"
(cd "$REPO/tools/ngram/tests" && NGRAM_LOOKUP_CHECK="$T/ngram_lookup_check" python3 -m unittest -q test_ngram.ExportTest)
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_dflreuse_test.cpp" -o "$T/batch_dflreuse_test" "${KNOBS[@]}"
"$T/batch_dflreuse_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_insert_test.cpp" -o "$T/batch_insert_test" "${KNOBS[@]}"
"$T/batch_insert_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_kstop_test.cpp" -o "$T/batch_kstop_test" "${KNOBS[@]}"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_draftpar_test.cpp" -o "$T/batch_draftpar_test" "${KNOBS[@]}"
"$T/batch_draftpar_test"
"$T/batch_kstop_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_dselmodel_test.cpp" -o "$T/batch_dselmodel_test" "${KNOBS[@]}"
"$T/batch_dselmodel_test"
python3 "$REPO/tools/s2_dsel_table.py" --selftest
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_kstopb_test.cpp" -o "$T/batch_kstopb_test" "${KNOBS[@]}"
"$T/batch_kstopb_test" 2>/dev/null
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/batch_kstop_mtp_test.cpp" -o "$T/batch_kstop_mtp_test" "${KNOBS[@]}"
"$T/batch_kstop_mtp_test"
g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/capture_barrier_test.cpp" -o "$T/capture_barrier_test" "${KNOBS[@]}"
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
# the scheduler's own prefill choice (fewest prompt rows first, with a deferral bound): no pool, no CUDA
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/scheduler_fair_test.cpp" -o "$T/scheduler_fair_test" "${KNOBS[@]}"
"$T/scheduler_fair_test"
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/scheduler_r_test.cpp" -o "$T/scheduler_r_test" "${KNOBS[@]}"
"$T/scheduler_r_test"
# the notice target's sizing (the work in flight, not a constant): no pool, no CUDA
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/scheduler_horizon_test.cpp" -o "$T/scheduler_horizon_test" "${KNOBS[@]}"
"$T/scheduler_horizon_test"
# Config::decode_rounds (GR8_S2_DECODE_ROUNDS): K decode rounds per prefill quantum, its notice target, default off
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/scheduler_rounds_test.cpp" -o "$T/scheduler_rounds_test" "${KNOBS[@]}"
"$T/scheduler_rounds_test"
# Config::tail_first (GR8_S2_TAIL_FIRST): a ready prompt tail before any prefill, no decode round forced after it
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/scheduler_tail_first_test.cpp" -o "$T/scheduler_tail_first_test" "${KNOBS[@]}"
"$T/scheduler_tail_first_test"
# the Driver's prefill quanta: the shortest prompt first, owed quanta, decode rounds between them
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/driver_prefill_quantum_test.cpp" -o "$T/driver_prefill_quantum_test" "${KNOBS[@]}"
"$T/driver_prefill_quantum_test" > /dev/null
# KvTiers::run's copy batches (s2/kv_plan.h): a batch of 1 advances, a slot spilled twice ends with the second page,
# batched copies = one at a time: no pool, no CUDA
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/kv_plan_test.cpp" -o "$T/kv_plan_test" "${KNOBS[@]}"
"$T/kv_plan_test"
# GR8_CP_STEP_ROWS / GR8_CP_STEP_SEG (s2/cp_rows.h, Driver::step): steps past R capped at n rows, solo-exact tokens, fair cycles
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/cp_rows_test.cpp" -o "$T/cp_rows_test" "${KNOBS[@]}"
"$T/cp_rows_test" 2> >(grep -v '^# s2 GR8_CP_STEP_ROWS=' >&2)   # the driver's cap transition lines, thousands here
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_stash_test.cpp" "${KVLIB[@]}" -o "$T/pool_stash_test"
STORES=("mem:" "ssd:$T/store")
NODE=$REPO/engine/kvman-rs/target/release/kvman-remote
if [ -x "$NODE" ]; then   # a small unlocked node on loopback (RLIMIT_MEMLOCK in a container is usually 8 MiB)
  "$NODE" --gb 0.05 --port 0 --bind 127.0.0.1 --min-free-gb 0 --no-lock --ready "$T/node.json" > "$T/node.log" 2>&1 &
  NODE_PID=$!
  for _ in $(seq 50); do [ -s "$T/node.json" ] && break; sleep 0.1; done
  PORT=$(sed -n 's/.*"port":\([0-9]*\).*/\1/p' "$T/node.json")
  [ -n "$PORT" ] && STORES+=("remote:127.0.0.1:$PORT")
  # the same node with --lazy-lock: each lease's range is backed only while it lives (it locks where RLIMIT_MEMLOCK
  # allows the lease, and runs unlocked otherwise)
  LOCK=--no-lock; [ "$(ulimit -l)" = unlimited ] || [ "$(ulimit -l)" -ge 131072 ] && LOCK=
  "$NODE" --gb 0.05 --port 0 --bind 127.0.0.1 --min-free-gb 0 $LOCK --lazy-lock --ready "$T/lazy.json" > "$T/lazy.log" 2>&1 &
  LAZY_PID=$!
  for _ in $(seq 50); do [ -s "$T/lazy.json" ] && break; sleep 0.1; done
  LPORT=$(sed -n 's/.*"port":\([0-9]*\).*/\1/p' "$T/lazy.json")
  [ -n "$LPORT" ] && STORES+=("remote:127.0.0.1:$LPORT")
fi
"$T/pool_stash_test" "${STORES[@]}"
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/pool_namespace_test.cpp" "${KVLIB[@]}" -o "$T/pool_namespace_test"
"$T/pool_namespace_test" "${STORES[@]}"
[ -n "${NODE_PID:-}" ] && kill "$NODE_PID"
[ -n "${LAZY_PID:-}" ] && kill "$LAZY_PID"
# a pool tier on a group lease that grows into the node's grant, gives a segment back when another cluster asks, and
# shrinks to its floor, through the C interface (s2/kv_share.h): a second node with 2 MiB segments and no margin
if [ -x "$NODE" ]; then
  g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/kv_share_test.cpp" "${KVLIB[@]}" -o "$T/kv_share_test"
  "$NODE" --gb 0.05 --seg-mb 2 --margin-s 0 --port 0 --bind 127.0.0.1 --min-free-gb 0 --no-lock --no-rdma --ready "$T/share.json" > "$T/share.log" 2>&1 &
  SHARE_PID=$!
  for _ in $(seq 50); do [ -s "$T/share.json" ] && break; sleep 0.1; done
  SPORT=$(sed -n 's/.*"port":\([0-9]*\).*/\1/p' "$T/share.json")
  "$T/kv_share_test" "remote:127.0.0.1:$SPORT"
  kill "$SHARE_PID"
fi
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime/s2" "$REPO/engine/runtime/s2/tests/control_protocol_test.cpp" -o "$T/control_protocol_test" "${KNOBS[@]}"
"$T/control_protocol_test"
# a held event boundary (Commit::hold): queued admissions chained at one boundary, three ranks on socket pairs
g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime/s2" "$REPO/engine/runtime/s2/tests/control_hold_test.cpp" -o "$T/control_hold_test" "${KNOBS[@]}"
"$T/control_hold_test"
# ranks of different control builds refuse at bootstrap (CONTROL_VERSION in the agreement and the frame magic)
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/control_version_test.cpp" -o "$T/control_version_test" "${KNOBS[@]}"
"$T/control_version_test"
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/steering_state_test.cpp" "${KVLIB[@]}" -o "$T/steering_state_test"
"$T/steering_state_test"
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/steering_transport_test.cpp" -o "$T/steering_transport_test" "${KNOBS[@]}"
"$T/steering_transport_test"
# the S2 trace (GR8_S2_TRACE): written by s2/trace.h, closed and joined by tools/trace/merge.py
g++ -std=c++17 -O2 -Wall -Wextra "$REPO/engine/runtime/s2/tests/trace_test.cpp" -o "$T/trace_test"
"$T/trace_test" "$T/trace"
python3 "$REPO/tools/trace/merge.py" "$T/trace.r3.json" -o "$T/trace.json" > /dev/null
python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$T/trace.json"
# merge.py --clock: a rank on another machine's clock moved onto the client's (tools/trace/clock_probe.py samples)
(cd "$REPO" && python3 -m unittest -q tools.trace.test_merge)
g++ -std=c++20 -O2 -Wall -pthread -I "$REPO/engine/loader" "$REPO/engine/loader/tests/manifest_test.cpp" \
  "$REPO/engine/loader/g6pack.cpp" -o "$T/manifest_test"
"$T/manifest_test"

g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/test_context_buckets.cpp" -o "$T/test_context_buckets" "${KNOBS[@]}"
"$T/test_context_buckets"

# the flags g6m4 and m4sim both take (engine/runtime/engine_opts.h)
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/engine_opts_test.cpp" -o "$T/engine_opts_test" "${KNOBS[@]}"
"$T/engine_opts_test"

# host memory after load (#4): the /proc parser and malloc_trim (glibc)
g++ -std=c++17 -O2 -Wall -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/host_reclaim_test.cpp" -o "$T/host_reclaim_test" "${KNOBS[@]}"
"$T/host_reclaim_test"

# Offline distributed logit dump framing and merge validation.
(cd "$REPO" && python3 -m unittest tools.cp.test_merge_logits)
# Flash KDA state rollback across accepted prefixes, and each session's slots and ring (s2/flash_ring.h).
g++ -std=c++17 -O2 -Wall "$REPO/engine/tests/test_flash_ring.cpp" -o "$T/test_flash_ring" "${KNOBS[@]}"
"$T/test_flash_ring"
# the KDA carry mode (Args::src per row: the ring, and several sessions in one step) on the CPU model of k_kda_core.
# Bit for bit across the kernel's M paths, so no FMA contraction (the host compiler would contract each path its own way).
g++ -std=c++17 -O2 -ffp-contract=off -DG6_MODEL_FLASH=1 "$REPO/engine/kda/tests/kda_carry_cpu_test.cpp" -o "$T/kda_carry_cpu_test" "${KNOBS[@]}"
"$T/kda_carry_cpu_test"
g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime/s2" "$REPO/engine/runtime/s2/tests/control_tp2_test.cpp" -o "$T/control_tp2_test" "${KNOBS[@]}"
"$T/control_tp2_test"
# FLASH_CAP_POLICY=borrow-idle's rule (s2/flash_lend.h) and the ACK's room (Ack::room): lend once idle, recall before a
# request that needs the chunks, the head admits within the least room over the ranks
g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime/s2" "$REPO/engine/runtime/s2/tests/flash_lend_test.cpp" -o "$T/flash_lend_test" "${KNOBS[@]}"
"$T/flash_lend_test"
# the Flash prefill checkpoint waits for the boundary's commits (s2/flash_native.cuh h.before_work): a grow announced
# during the prefill lands before the capture, on two ranks through the real Service loop
g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/flash_capture_order_test.cpp" "${KVLIB[@]}" -o "$T/flash_capture_order_test"
(cd "$T" && "$T/flash_capture_order_test" 2>/dev/null)
# GR8_S2_ADMIT_HOLD=1: ADMIT files that land inside a held admission join the held boundary
g++ -std=c++17 -O2 -Wall -pthread -I "$REPO/engine/runtime" "$REPO/engine/runtime/s2/tests/admit_hold_drain_test.cpp" "${KVLIB[@]}" -o "$T/admit_hold_drain_test"
(cd "$T" && "$T/admit_hold_drain_test" 2>/dev/null)
g++ -std=c++17 -O2 -Wall "$REPO/engine/tests/test_flash_s2_cost.cpp" -o "$T/test_flash_s2_cost" "${KNOBS[@]}"
"$T/test_flash_s2_cost"
g++ -std=c++17 -O2 -Wall -DG6_NR=3 "$REPO/engine/tests/test_flash_s2_cost.cpp" -o "$T/test_flash_s2_cost_tp3" "${KNOBS[@]}"
"$T/test_flash_s2_cost_tp3"
g++ -std=c++17 -O2 -Wall -DG6_MODEL_FLASH=0 "$REPO/engine/tests/test_flash_s2_cost.cpp" -o "$T/test_full_s2_cost" "${KNOBS[@]}"
"$T/test_full_s2_cost"
# DeepSeek-V4.1-Flash-1017 geometry (engine/model_geom_ds41.h) against the release's config.json
g++ -std=c++17 -O2 -Wall -DG6_MODEL_DS41=1 -I "$REPO/engine" "$REPO/engine/tests/ds41_geom_dump.cpp" -o "$T/ds41_geom_dump"
(cd "$REPO/engine/tests" && DS41_GEOM_DUMP="$T/ds41_geom_dump" python3 -m unittest -q test_ds41_geom)
python3 "$REPO/engine/tests/test_flash_serve_check.py"
# Flash serve prefill: the draft cache fill checks the error words once, after its last chunk; every error still fails the request.
python3 "$REPO/engine/tests/test_flash_serve_prefill_check.py"
# Flash prefill non-kernel span (NK): GR8_GLM53F_PF_PREWARM source check, GR8_GLM53F_PF_STAGE_ASYNC source check + schedule hazard model.
python3 -m unittest "$REPO/engine/runtime/tests/test_pf_prewarm.py" "$REPO/engine/runtime/tests/test_pf_stage_async.py"
g++ -std=c++17 -O2 -Wall "$REPO/engine/runtime/tests/pf_peer_stage_test.cpp" -o "$T/pf_peer_stage_test" "${KNOBS[@]}"
"$T/pf_peer_stage_test"
# Flash prefill router row split (GR8_PF_ROUTER_SPLIT): every rank count 2..8 ends with byte-equal route tables, the kernels'
# word maps included; the strict env parse; the source checks of its boot and order.
for t in test_flash_route_split test_flash_route_split_tp4 test_flash_route_split_env; do
  g++ -std=c++17 -O1 -Wall "$REPO/engine/tests/$t.cpp" -o "$T/$t" "${KNOBS[@]}"
  "$T/$t"
done
python3 "$REPO/engine/tests/test_flash_route_split_source.py"
# Flash session checkpoints (GR8_GLM53F_CACHE): aligned points, the index's match/LRU/first-fit, the manifest seal.
g++ -std=c++17 -O2 -Wall -Wextra -I "$REPO/engine/runtime" "$REPO/engine/tests/test_flash_ckpt_index.cpp" -o "$T/test_flash_ckpt_index" "${KNOBS[@]}"
"$T/test_flash_ckpt_index"
# GR8_GLM53F_CACHE=terminal: the actual retain hook against CPU mocks (rows, ring slot, tokens, nothing kept).
python3 "$REPO/engine/tests/test_flash_terminal_retain.py"
# GR8_GLM53F_CACHE=terminal, a cancel at the boundary the aligned capture waits for: flash_native.cuh's own hooks through
# the real Service on two ranks; the entry stays aligned and a longer prompt keeps its own aligned checkpoint
python3 "$REPO/engine/tests/test_flash_terminal_cancel.py"
# the adaptive draft length decision, and the top-k selection
g++ -std=c++17 -O2 "$REPO/engine/tests/test_spec_len.cpp" -o "$T/test_spec_len" "${KNOBS[@]}"
"$T/test_spec_len"
g++ -std=c++17 -O2 -I "$REPO/engine/kernels" "$REPO/engine/tests/test_topk.cpp" -o "$T/test_topk" "${KNOBS[@]}"
"$T/test_topk"
# the control channel: requests, sync, and idle peers (the socketpair cases; the silent-peer case needs a network
# namespace)
for t in control_request_test control_sync_test control_idle_test; do
  g++ -std=c++17 -O2 -pthread -I "$REPO/engine/runtime/s2" "$REPO/engine/runtime/s2/tests/$t.cpp" -o "$T/$t" "${KNOBS[@]}"
  "$T/$t"
done
g++ -std=c++17 -O1 -fsanitize=address,undefined -I "$REPO/engine/runtime" "$REPO/engine/runtime/tests/serve_parse_test.cpp" \
  -o "$T/serve_parse_test"
"$T/serve_parse_test"
# the fabric's progress CPU lists; needs 3 or more CPUs
g++ -std=c++20 -O1 -pthread -I "$REPO/engine/comms" "$REPO/tools/health/test_progress_cpus.cpp" -o "$T/test_progress_cpus" "${KNOBS[@]}"
"$T/test_progress_cpus"
# CP batching on the CPU, the Flash DSA full-attention schedule, the Flash pool owner's rows and TP2 wire format, the
# MoE top-k env parse, the Flash per-request speculation reset
g++ -std=c++17 -O2 "$REPO/engine/runtime/tests/cp_batch_cpu_test.cpp" -o "$T/cp_batch_cpu_test" "${KNOBS[@]}"
"$T/cp_batch_cpu_test"
g++ -std=c++17 -O2 "$REPO/engine/runtime/tests/flash_dsa_full_schedule_test.cpp" -o "$T/flash_dsa_full_schedule_test" "${KNOBS[@]}"
"$T/flash_dsa_full_schedule_test"
g++ -std=c++17 -O2 "$REPO/engine/tests/test_flash_pool_owner.cpp" -o "$T/test_flash_pool_owner" "${KNOBS[@]}"
"$T/test_flash_pool_owner"
g++ -std=c++17 -O2 -I "$REPO/engine/runtime" "$REPO/engine/tests/test_moe_topk_env.cpp" -o "$T/test_moe_topk_env" "${KNOBS[@]}"
"$T/test_moe_topk_env"
g++ -std=c++17 -O2 -DG6_MODEL_FLASH=1 "$REPO/engine/tests/test_spec_reset.cpp" -o "$T/test_spec_reset" "${KNOBS[@]}"
"$T/test_spec_reset"
# the glue kda_g9_test.cu feeds the KDA G9 CPU emulation with (kda_g9_glue.h), on two heads
g++ -std=c++17 -O3 -fopenmp -DG6_MODEL_FLASH=1 "$REPO/engine/kda/tests/kda_g9_glue_test.cpp" -o "$T/kda_g9_glue_test" "${KNOBS[@]}"
"$T/kda_g9_glue_test"
# the bare environment reads the knob table replaced, and the table against the tree
python3 "$REPO/tools/env_lint.py"
