#!/bin/bash
# Build (if needed) and run the unit tests on ONE GPU node (a GB10 Spark with CUDA 13): row_budget_test (dense GEMVs and K1
# against chunks of 8 rows), s2_attn_test (batched attention over several sessions against the per-session calls;
# needs EXTRA=-DDEC_S2, else it only prints SKIP), dec_heads_test (the decode attention kernels against the ones they
# replaced) and kv4_test (the KV latent row: append round trip, and decode / prefill / row-owner attention against a host
# reference; EXTRA=-DG6_KV4=1 tests the INT4 row, docs/KV4.md) and weight_check_test (the weight check's GPU sums
# against the CPU's, and one changed byte caught in its own chunk) and canon_test (a session's KV exported and imported
# across page lists and layouts, then a decode step on it; #152) and kv_resize_test (the KV pool's chunks shrunk and grown
# under live sessions, on caches in virtual memory, against caches that never resized; #153) and dhead_i4_test
# (G6_DRAFT_HEAD_I4: the MXFP8 -> INT4 head records against a host derivation, the INT4 head GEMV against a host dot;
# --time: MXFP8 vs INT4 head GEMV) and kvb8_test (kv_b in FP8: decode and prefill against the BF16 form of the same
# weights, byte for byte; --bench times absorb / value_project; #50).
#   scripts/c1_tests.sh <bin_dir> [test args]       default bin_dir = ./bin; TEST=row_budget_test|s2_attn_test runs one
#   G6_NR=4 scripts/c1_tests.sh bin4                the TP4 build of the test
#   EXTRA=-DDEC_S2 scripts/c1_tests.sh bin_s2       the 32-row build: M = 1..32 against chunks of 8 (without it: M <= 8)
#   ARCH=120a ...                                   RTX 50xx instead of the GB10 (default 121a)
# Compiles engine/runtime/tests/<test>.cu with the K1H2 flags of scripts/build.sh into <bin_dir>/<test> (s2_attn_test also
# compiles attn/attn.cu and attn/bind.cu into it), skipping a compile when the binary is newer than the sources and was built
# with the same flags.
# Exit code: 0 = every case PASS, 1 = a case FAILED, 2 = CUDA error, other = build failure (the first failing test's code).
# Test args go to the selected test (TEST=...), with both selected to row_budget_test:
#   row_budget_test: --only gv|k1, --time [reps] (K1 timing over M = 6..24)
#   s2_attn_test:    --only rep|cp, --seeds N, --no-long (skips the cases above 32768 tokens: two replicated, two CP)
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$REPO/bin}" && cd "${1:-$REPO/bin}" && pwd)
shift || true
export PATH=/usr/local/cuda/bin:$PATH
A=${ARCH:-121a}
# keep in step with K1H2_FLAGS in scripts/build.sh (DEC_SLOT=2 and DEC_I4PAIR=1 are the paths the engine ships)
K1H2_FLAGS="-DPF_FC2GL=6 -DPF_GEMMF=1 -DPF_XQ8C=1 -DPF_RNWIDE=1 -DPF_GEMMR=1 -DPF_DRAFTKV -DPF_GEMMRF=1 -DPF_QOBH=1 \
-DDEC_SLOT=2 -DDEC_CPB=1 -DDEC_AVP=1 -DPF_QOROPE=1 -DPF_ISGRT=1 -DPF_ROPE2=1 -DDEC_CPB2=64 -DDEC_I4PAIR=1 -DSEL32 \
-DG6_CP=1 -DG6_CP_PF_HASH2=1 -DDEC_IDX31=1 -DPF_QO_DEDUP=1"
FLAGS="-gencode=arch=compute_$A,code=sm_$A $K1H2_FLAGS ${G6_NR:+-DG6_NR=$G6_NR} ${EXTRA:-}"
build_run() {   # build_run <test> <extra sources...>  (test args in ARGS)
  local name=$1; shift
  local SRC="$REPO/engine/runtime/tests/$name.cu" BIN="$OUT/$name"
  local stale=0
  [ -x "$BIN" ] || stale=1
  [ "$(cat "$BIN.flags" 2>/dev/null)" = "$FLAGS" ] || stale=1
  if [ "$stale" = 0 ] && [ -n "$(find "$REPO/engine" -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.h' \) -newer "$BIN" -print -quit)" ]; then stale=1; fi
  local a; for a in "$@"; do [ -f "$a" ] && [ "$a" -nt "$BIN" ] && stale=1; done   # e.g. a rebuilt libg6kvman.a
  if [ "$stale" = 1 ]; then
    echo "building $BIN ($FLAGS)"
    rm -f "$BIN" "$BIN.flags"
    # shellcheck disable=SC2086
    nvcc -O2 -std=c++17 $FLAGS -o "$BIN" "$SRC" "$@"
    echo "$FLAGS" > "$BIN.flags"
  else
    echo "up to date: $BIN"
  fi
  local rc=0
  "$BIN" ${ARGS[@]+"${ARGS[@]}"} || rc=$?
  return $rc
}
# kvlib_ready: target/release/libg6kvman.a for the tests that link it, built first when cargo is on PATH; 1 (with
# the reason) when it is missing or older than the crate's sources
kvlib_ready() {
  local crate="$REPO/engine/kvman-rs" lib="$REPO/engine/kvman-rs/target/release/libg6kvman.a"
  if command -v cargo > /dev/null; then
    cargo build --release -q --manifest-path "$crate/Cargo.toml" || { echo "cargo build --release failed"; return 1; }
  fi
  [ -f "$lib" ] || { echo "no $lib (cargo build --release --manifest-path engine/kvman-rs/Cargo.toml)"; return 1; }
  if [ -n "$(find "$crate/src" "$crate/Cargo.toml" -newer "$lib" -print -quit)" ]; then
    echo "$lib is older than engine/kvman-rs/src (cargo build --release --manifest-path engine/kvman-rs/Cargo.toml)"
    return 1
  fi
}
TEST=${TEST:-all}
ARGS=("$@")
rc=0
if [ "$TEST" = all ] || [ "$TEST" = row_budget_test ]; then
  build_run row_budget_test || rc=$?
fi
if [ "$TEST" = all ] || [ "$TEST" = s2_attn_test ]; then
  [ "$TEST" = all ] && ARGS=()
  case "$FLAGS" in
    *-DDEC_S2*) build_run s2_attn_test "$REPO/engine/attn/attn.cu" "$REPO/engine/attn/bind.cu" "$REPO/engine/attn/ops_glm.cu" -lcublas || { r=$?; [ "$rc" = 0 ] && rc=$r; } ;;
    *) build_run s2_attn_test || { r=$?; [ "$rc" = 0 ] && rc=$r; } ;;
  esac
fi
if [ "$TEST" = all ] || [ "$TEST" = dec_heads_test ]; then
  [ "$TEST" = all ] && ARGS=()
  case "$FLAGS" in
    *-DG6_KV4=1*) echo "SKIP dec_heads_test: it checks the FP8-row kernels, which a G6_KV4 build does not run" ;;
    *) build_run dec_heads_test "$REPO/engine/attn/bind.cu" -lcublas || { r=$?; [ "$rc" = 0 ] && rc=$r; } ;;
  esac
fi
if [ "$TEST" = all ] || [ "$TEST" = kv4_test ]; then
  [ "$TEST" = all ] && ARGS=()
  build_run kv4_test "$REPO/engine/attn/attn.cu" "$REPO/engine/attn/bind.cu" "$REPO/engine/attn/ops_glm.cu" -lcublas || { r=$?; [ "$rc" = 0 ] && rc=$r; }
fi
if [ "$TEST" = all ] || [ "$TEST" = kvb8_test ]; then
  # kv_b (UK / UV) in FP8 (#50): decode and prefill outputs against the BF16 form of the same weights, byte for byte
  [ "$TEST" = all ] && ARGS=()
  case "$FLAGS" in
    *-DG6_KV4=1*) echo "SKIP kvb8_test: it runs the FP8-row decode variants, which a G6_KV4 build does not" ;;
    *) build_run kvb8_test "$REPO/engine/attn/bind.cu" -lcublas || { r=$?; [ "$rc" = 0 ] && rc=$r; } ;;
  esac
fi
if [ "$TEST" = all ] || [ "$TEST" = canon_test ]; then
  # a session's KV across page lists and layouts (#152): links the S2 page allocator's library (engine/kvman-rs)
  [ "$TEST" = all ] && ARGS=()
  KVLIB="$REPO/engine/kvman-rs/target/release/libg6kvman.a"
  if why=$(kvlib_ready); then
    build_run canon_test "$REPO/engine/attn/attn.cu" "$REPO/engine/attn/bind.cu" "$REPO/engine/attn/ops_glm.cu" "$KVLIB" -lcublas -lpthread -ldl || { r=$?; [ "$rc" = 0 ] && rc=$r; }
  else
    echo "SKIP canon_test: $why"
  fi
fi
if [ "$TEST" = all ] || [ "$TEST" = kv_resize_test ]; then
  # the KV pool's chunks shrunk and grown under live sessions (#153): the allocator's library and the driver API
  [ "$TEST" = all ] && ARGS=()
  KVLIB="$REPO/engine/kvman-rs/target/release/libg6kvman.a"
  if why=$(kvlib_ready); then
    build_run kv_resize_test "$REPO/engine/attn/attn.cu" "$REPO/engine/attn/bind.cu" "$REPO/engine/attn/ops_glm.cu" "$REPO/engine/attn/kvchunks.cu" "$KVLIB" -lcublas -lcuda -lpthread -ldl || { r=$?; [ "$rc" = 0 ] && rc=$r; }
  else
    echo "SKIP kv_resize_test: $why"
  fi
fi
if [ "$TEST" = all ] || [ "$TEST" = flash_kv_chunks_test ]; then
  # the Flash KV caches in chunks (G6_FLASH_KV_CHUNKS, runtime/flash_kv.h): the Flash attention, the ledger's library
  [ "$TEST" = all ] && ARGS=()
  KVLIB="$REPO/engine/kvman-rs/target/release/libg6kvman.a"
  if why=$(kvlib_ready); then
    GLM_FLAGS="$FLAGS"; FLAGS="$FLAGS -DG6_MODEL_FLASH=1"
    KVMAN_LEDGER="$REPO/engine/kvman-rs/target/release/kvman-ledger" build_run flash_kv_chunks_test "$REPO/engine/attn/flash_attn.cu" "$REPO/engine/attn/flash_bind.cu" "$REPO/engine/attn/ops_flash.cu" "$REPO/engine/attn/kvchunks.cu" "$KVLIB" -lcuda -lpthread -ldl || { r=$?; [ "$rc" = 0 ] && rc=$r; }
    FLAGS="$GLM_FLAGS"
  else
    echo "SKIP flash_kv_chunks_test: $why"
  fi
fi
if [ "$TEST" = all ] || [ "$TEST" = flash_ckpt_group_test ]; then
  # Flash checkpoints on a group lease (D-331 M8 (c), s2/flash_checkpoint.cuh): a loopback KV node with 2 MiB segments
  [ "$TEST" = all ] && ARGS=()
  KVLIB="$REPO/engine/kvman-rs/target/release/libg6kvman.a"
  NODE="$REPO/engine/kvman-rs/target/release/kvman-remote"
  if why=$(kvlib_ready) && [ -x "$NODE" ]; then
    NJ="$OUT/flash_ckpt_group_node.json"
    : > "$NJ"
    "$NODE" --gb 0.05 --seg-mb 2 --margin-s 0 --port 0 --bind 127.0.0.1 --min-free-gb 0 --no-lock --no-rdma --ready "$NJ" \
      > "$OUT/flash_ckpt_group_node.log" 2>&1 &
    NODE_PID=$!
    for _ in $(seq 50); do [ -s "$NJ" ] && break; sleep 0.1; done
    [ ${#ARGS[@]} -gt 0 ] || ARGS=("remote:127.0.0.1:$(sed -n 's/.*"port":\([0-9]*\).*/\1/p' "$NJ")")
    build_run flash_ckpt_group_test "$KVLIB" -lpthread -ldl -lm || { r=$?; [ "$rc" = 0 ] && rc=$r; }
    kill "$NODE_PID"
  else
    echo "SKIP flash_ckpt_group_test: ${why:-no kvman-remote: cargo build --release}"
  fi
fi
if [ "$TEST" = all ] || [ "$TEST" = weight_check_test ]; then
  [ "$TEST" = all ] && ARGS=()
  build_run weight_check_test || { r=$?; [ "$rc" = 0 ] && rc=$r; }
fi
if [ "$TEST" = all ] || [ "$TEST" = steering_boot_cuda_test ]; then
  build_run steering_boot_cuda_test -lcublas || { r=$?; [ "$rc" = 0 ] && rc=$r; }
fi
if [ "$TEST" = all ] || [ "$TEST" = steering_shared_test ]; then
  build_run steering_shared_test -lcublas || { r=$?; [ "$rc" = 0 ] && rc=$r; }
fi
if [ "$TEST" = all ] || [ "$TEST" = steering_consume_test ]; then
  [ "$TEST" = all ] && ARGS=()
  build_run steering_consume_test -lcublas || { r=$?; [ "$rc" = 0 ] && rc=$r; }
fi
if [ "$TEST" = all ] || [ "$TEST" = steering_cuda_test ]; then
  [ "$TEST" = all ] && ARGS=()
  build_run steering_cuda_test || { r=$?; [ "$rc" = 0 ] && rc=$r; }
fi
if [ "$TEST" = all ] || [ "$TEST" = dhead_i4_test ]; then
  [ "$TEST" = all ] && ARGS=()
  build_run dhead_i4_test || { r=$?; [ "$rc" = 0 ] && rc=$r; }
fi
if [ "$rc" = 0 ]; then echo "c1_tests: PASS"; else echo "c1_tests: FAIL (exit $rc)"; fi
exit "$rc"
