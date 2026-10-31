#!/bin/bash
# Build the K1H2 engine (g6m4 = one rank on the real fabric, m4sim = one GPU simulating all NR ranks) on ONE Spark.
# Needs CUDA 13 (nvcc with sm_121a), g++ with C++20, libibverbs-dev, cuBLAS. A few minutes on a GB10, CPU only.
# One g6m4 and one m4sim run every TP width: g6m4 takes its width from --ips, m4sim from --tp N or --packs DIR. Each
# width is compiled as its own build was (engine/runtime/build_m4.sh): DEC_S2 (the multi-session path) at every width,
# and PF_F12 (the fat-CTA prefill fc1) at 4.
#   scripts/build.sh [OUT_DIR]            default OUT_DIR = ./bin; TP = 1, 2, 3, 4, 5, 6 and 8
#   GR8_WIDTHS="3 4" scripts/build.sh bin only those widths (a shorter build); GR8_NR=4 is GR8_WIDTHS=4
#   F12_WIDTHS="4 5" scripts/build.sh bin  PF_F12 (fat-CTA prefill fc1, on unless GR8_PF_F12=0) at TP5 too; default TP4 only
#   JOBS=N                                compiles at once (default: the CPU count)
#   KV_FORMAT=int4 scripts/build.sh bin   only OUT_DIR/g6m4-int4: the engine with the 4-bit KV latent (-DG6_KV4=1,
#                                         docs/KV4.md), which `--kv-format int4` runs. The same widths.
#   MODEL=flash scripts/build.sh bin      only OUT_DIR/g6m4_flash and OUT_DIR/m4sim_flash: GLM 5.3 Flash
#                                         (-DG6_MODEL_FLASH=1, engine/model_geom.h), fp8 KV only; TP = 2 and 4 unless
#                                         GR8_WIDTHS or GR8_NR says otherwise
#   MODEL=ds41 scripts/build.sh bin       only OUT_DIR/g6m4_ds41 and OUT_DIR/m4sim_ds41: DeepSeek-V4.1-Flash-1017
#                                         (-DG6_MODEL_DS41=1, engine/model_geom_ds41.h, docs/DEEPSEEK.md), fp8 KV only; TP = 4
#                                         unless GR8_WIDTHS or GR8_NR says otherwise
#   GR8_TESTS=0 / GR8_TESTS=only        the GLM engine without the test binaries below it, or the tests alone (CI
#                                         builds them on separate runners)
#   BUNDLE_LIBS=1                         also copy cuBLAS (libcublas, libcublasLt) into OUT_DIR/lib, for nodes
#                                         without the CUDA toolkit (the engine finds it there at any TP)
# These -D flags are the K1H2 build; every one of them is an engine path that our A/B tests promoted
# (see docs/SETTINGS.md). Dropping one builds an older, slower or less-checked path, not a broken one.
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$REPO/bin}" && cd "${1:-$REPO/bin}" && pwd)
K1H2_FLAGS="-DPF_FC2GL=6 -DPF_GEMMF=1 -DPF_XQ8C=1 -DPF_RNWIDE=1 -DPF_GEMMR=1 -DPF_DRAFTKV -DPF_GEMMRF=1 -DPF_QOBH=1 \
-DDEC_SLOT=2 -DDEC_CPB=1 -DDEC_AVP=1 -DPF_QOROPE=1 -DPF_ISGRT=1 -DPF_ROPE2=1 -DDEC_CPB2=64 -DDEC_I4PAIR=1 -DSEL32 \
-DG6_CP=1 -DG6_CP_PF_HASH2=1 -DDEC_IDX31=1 -DPF_QO_DEDUP=1"
# the Rust pack loader (engine/loader-rs), and bin/g6load, which plans and checks the Hugging Face fetch. Where cargo is
# missing (the CUDA image), build it on the host first: cargo build --release --manifest-path engine/loader-rs/Cargo.toml
# (g6m4 links target/release/libg6load.a when it exists, and target/release/g6load is copied next to g6m4)
# The S2 page allocator (engine/kvman-rs) and the knob table's C API (engine/knobs-rs) are not optional: g6m4 always
# links target/release/libg6kvman.a and libgr8knobs.a, built the same way, and every test that reads a knob links the
# second.
if command -v cargo > /dev/null 2>&1; then
  cargo build --release --quiet --manifest-path "$REPO/engine/loader-rs/Cargo.toml"
  cargo build --release --quiet --manifest-path "$REPO/engine/kvman-rs/Cargo.toml"
  cargo build --release --quiet --features capi --manifest-path "$REPO/engine/knobs-rs/Cargo.toml"
fi
KNOBS=("$REPO/engine/knobs-rs/target/release/libgr8knobs.a" -lpthread -ldl -lm)
# the old G6_ names of these variables still work
for v in NR WIDTHS TESTS XGRAMMAR XGRAMMAR_LIB NVCC_CACHE NVCC_CACHE_DIR; do
  o=G6_$v n=GR8_$v; if [ -n "${!o:-}" ] && [ -z "${!n:-}" ]; then export "$n=${!o}"; fi
done
# request grammars (engine/runtime/grammar.h): XGrammar's C++, the copy the front end's xgrammar crate pins
# (frontend/g6front-rs/Cargo.lock), so the engine's masks and the front end's checks run one grammar library.
# GR8_XGRAMMAR=DIR names a source tree instead (the image's engine stages), GR8_XGRAMMAR=0 builds without grammars.
# Needs cmake; GLM-5.3 only
GR_OFF=""
if [ "${MODEL:-glm}" = flash ]; then GR_OFF="MODEL=flash"
elif [ "${MODEL:-glm}" = ds41 ]; then GR_OFF="MODEL=ds41"
elif [ "${GR8_TESTS:-1}" = only ]; then GR_OFF="GR8_TESTS=only builds no engine"
elif [ "${GR8_XGRAMMAR:-}" = 0 ]; then GR_OFF="GR8_XGRAMMAR=0"
fi
if [ -n "$GR_OFF" ]; then
  unset GR8_XGRAMMAR
elif [ -z "${GR8_XGRAMMAR:-}" ] && command -v cargo > /dev/null 2>&1 && command -v cmake > /dev/null 2>&1; then
  F=$REPO/frontend/g6front-rs/Cargo.toml
  cargo fetch --quiet --manifest-path "$F" 2> /dev/null || true
  M=$(cargo metadata --quiet --format-version 1 --manifest-path "$F" 2> /dev/null |
      grep -o '"manifest_path":"[^"]*/xgrammar-[0-9][^"/]*/Cargo.toml"' | head -1 | sed 's/.*:"//; s/"$//')
  [ -n "$M" ] && [ -d "$(dirname "$M")/thirdparty/xgrammar" ] && GR8_XGRAMMAR=$(dirname "$M")/thirdparty/xgrammar
fi
if [ -n "${GR8_XGRAMMAR:-}" ]; then export GR8_XGRAMMAR; echo "request grammars: XGrammar from $GR8_XGRAMMAR"
elif [ -n "$GR_OFF" ]; then echo "request grammars: off ($GR_OFF)"
else echo "request grammars: off (no XGrammar source; it needs cargo and cmake, or GR8_XGRAMMAR=DIR)"; fi
G6LOAD_BIN=$REPO/engine/loader-rs/target/release/g6load
if [ -x "$G6LOAD_BIN" ]; then cp "$G6LOAD_BIN" "$OUT/"; fi
# the knob table the launcher reads (tools/launch/knobs.py)
KNOBSO=$REPO/engine/knobs-rs/target/release/libgr8knobs.so
if [ -f "$KNOBSO" ]; then mkdir -p "$OUT/lib" && cp "$KNOBSO" "$OUT/lib/"; fi
# the KV node ./gr8 kv up runs
KVNODE_BIN=$REPO/engine/kvman-rs/target/release/kvman-remote
if [ -x "$KVNODE_BIN" ]; then cp "$KVNODE_BIN" "$OUT/"; fi
# the host ledger of a Flash box that lends memory to a KV node (FLASH_KV_DONATE_GIB)
LEDGER_BIN=$REPO/engine/kvman-rs/target/release/kvman-ledger
if [ -x "$LEDGER_BIN" ]; then cp "$LEDGER_BIN" "$OUT/"; fi
cd "$REPO/engine/runtime"
[ "${PLAIN:-0}" = 1 ] && { echo "PLAIN=1 is retired: every width has the multi-session path (DEC_S2)"; exit 2; }
WIDTHS=${GR8_WIDTHS:-${GR8_NR:-1 2 3 4 5 6 8}}
# each width runs its own instance: the --top-p-cands bound is TP_SORT (2048) / NR, checked before any CUDA work
widths_ok() {   # widths_ok <g6m4> [<m4sim>]
  local n ips want
  for n in $WIDTHS; do
    ips=$(printf 'x,%.0s' $(seq "$n")); want="--top-p-cands must be 0..$((2048 / n))"
    "$1" --rank 0 --ips "${ips%,}" --manifest m --pack p --top-p-cands 99999 2>&1 | grep -qF -- "$want" \
      || { echo "$1 does not run TP=$n"; exit 1; }
    if [ -n "${2:-}" ]; then
      "$2" --tp "$n" --packs . --top-p-cands 99999 2>&1 | grep -qF -- "$want" || { echo "$2 does not run TP=$n"; exit 1; }
    fi
  done
  echo "$(basename "$1")${2:+ and $(basename "$2")} run TP = $WIDTHS"
}
case "${KV_FORMAT:-fp8}" in
  fp8) ;;
  int4)   # g6m4 alone, in its own object dir, installed as g6m4-int4 next to the fp8 build
    WIDTHS="$WIDTHS" EXTRA="$K1H2_FLAGS -DG6_KV4=1 ${EXTRA:-}" OUT="$OUT/int4" bash build_m4.sh "${ARCH:-121a}" g6
    mv "$OUT/int4/g6m4" "$OUT/g6m4-int4" && rm -rf "$OUT/int4"
    strings "$OUT/g6m4-int4" | grep -q int4g32h || { echo "g6m4-int4: not the int4 build (no int4g32h row name)"; exit 1; }
    md5sum "$OUT/g6m4-int4"
    widths_ok "$OUT/g6m4-int4"
    exit 0 ;;
  *) echo "KV_FORMAT=$KV_FORMAT: fp8 or int4"; exit 2 ;;
esac
case "${MODEL:-glm}" in
  glm) ;;
  flash)   # g6m4 and m4sim in their own object dir, installed as g6m4_flash / m4sim_flash next to the GLM build
    [ "${KV_FORMAT:-fp8}" = fp8 ] || { echo "MODEL=flash: KV_FORMAT=fp8 only"; exit 2; }
    WIDTHS=${GR8_WIDTHS:-${GR8_NR:-2 3 4}}
    # PF_FC2GL must divide HID / 128 = 32 (GLM 48: 6); build_m4.sh leaves out PF_F12, which the Flash fp16 MoE refuses
    WIDTHS="$WIDTHS" EXTRA="${K1H2_FLAGS/-DPF_FC2GL=6/-DPF_FC2GL=4} -DG6_MODEL_FLASH=1 ${EXTRA:-}" OUT="$OUT/flash" \
      bash build_m4.sh "${ARCH:-121a}" all
    mv "$OUT/flash/m4sim" "$OUT/m4sim_flash" && mv "$OUT/flash/g6m4" "$OUT/g6m4_flash" && rm -rf "$OUT/flash"
    md5sum "$OUT/m4sim_flash" "$OUT/g6m4_flash"
    widths_ok "$OUT/g6m4_flash" "$OUT/m4sim_flash"
    # the Flash serve's own hooks are in the binary (borrow-idle lived in GLM-only code once and no unit test saw it)
    for m in "flash kv: borrow-idle, %u of" "flash ckpt share: now"; do
      grep -c -a -F "$m" "$OUT/g6m4_flash" > /dev/null || { echo "$OUT/g6m4_flash lacks '$m'"; exit 1; }
    done
    echo "g6m4_flash has its serve hooks"
    exit 0 ;;
  ds41)    # g6m4 and m4sim in their own object dir, installed as g6m4_ds41 / m4sim_ds41 next to the GLM build
    [ "${KV_FORMAT:-fp8}" = fp8 ] || { echo "MODEL=ds41: KV_FORMAT=fp8 only"; exit 2; }
    WIDTHS=${GR8_WIDTHS:-${GR8_NR:-4}}
    # PF_FC2GL must divide HID / 128 = 40 (GLM 48: 6; Flash 32: 4; 4 divides 40); build_m4.sh leaves out PF_F12 for every non-GLM model
    # DS41_WHAT=sim: m4sim only (first light: no g6m4, so no libibverbs and no fabric objects)
    WIDTHS="$WIDTHS" EXTRA="${K1H2_FLAGS/-DPF_FC2GL=6/-DPF_FC2GL=4} -DG6_MODEL_DS41=1 ${EXTRA:-}" OUT="$OUT/ds41" \
      bash build_m4.sh "${ARCH:-121a}" "${DS41_WHAT:-all}"
    if [ "${DS41_WHAT:-all}" = sim ]; then
      mv "$OUT/ds41/m4sim" "$OUT/m4sim_ds41" && rm -rf "$OUT/ds41"
      md5sum "$OUT/m4sim_ds41"
      exit 0
    fi
    mv "$OUT/ds41/m4sim" "$OUT/m4sim_ds41" && mv "$OUT/ds41/g6m4" "$OUT/g6m4_ds41" && rm -rf "$OUT/ds41"
    md5sum "$OUT/m4sim_ds41" "$OUT/g6m4_ds41"
    widths_ok "$OUT/g6m4_ds41" "$OUT/m4sim_ds41"
    exit 0 ;;
  *) echo "MODEL=$MODEL: glm, flash or ds41"; exit 2 ;;
esac
case "${GR8_TESTS:-1}" in 0|1|only) ;; *) echo "GR8_TESTS=$GR8_TESTS: 0, 1 or only"; exit 2 ;; esac
if [ "${GR8_TESTS:-1}" != only ]; then
  WIDTHS="$WIDTHS" EXTRA="$K1H2_FLAGS ${EXTRA:-}" OUT="$OUT" bash build_m4.sh "${ARCH:-121a}" all
  if [ "${BUNDLE_LIBS:-0}" = 1 ]; then
    mkdir -p "$OUT/lib"
    cp -L /usr/local/cuda/lib64/libcublas.so.13 /usr/local/cuda/lib64/libcublasLt.so.13 "$OUT/lib/"
  fi
  strings "$OUT/g6m4" | grep -q "G6_CP_PF_HASH2 build" && echo "HASH2 build marker: present"
  md5sum "$OUT/g6m4" "$OUT/m4sim"
  widths_ok "$OUT/g6m4" "$OUT/m4sim"
fi
[ "${GR8_TESTS:-1}" = 0 ] && exit 0
export PATH=/usr/local/cuda/bin:$PATH
# the compiles below go through the same object cache as build_m4.sh (#32); GR8_NVCC_CACHE=0 runs plain nvcc / g++
CACHE="$REPO/engine/runtime/nvcc_cache.sh"
if [ "${GR8_NVCC_CACHE:-1}" = 0 ]; then NVCC=nvcc; GXX=g++; else NVCC="$CACHE nvcc"; GXX="$CACHE g++"; fi
mkdir -p "$OUT/obj"
export GR8_NVCC_CACHE_LOG="$OUT/obj/nvcc_cache.tests.calls"
rm -f "$GR8_NVCC_CACHE_LOG"
GEN="-gencode=arch=compute_${ARCH:-121a},code=sm_${ARCH:-121a}"
S2="$GEN $K1H2_FLAGS -DG6_NR=4 -DDEC_S2"
TL="$OUT/obj/tests"
rm -rf "$TL" && mkdir -p "$TL"
J=${JOBS:-$(nproc)}
build() {   # build <bin> <command..>: in the background, at most J at once; a failure leaves $TL/<bin>.failed
  while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n || true; done
  { "${@:2}" > "$TL/$1.log" 2>&1 || touch "$TL/$1.failed"; } &
}
# The compiles run first, the longest first. The tests run after them, in order.
# GR8_L2PF (paced side-stream L2 prefetch in the step's idle-DRAM windows): byte-exact check + per-layer timing of the
# decode layer sequence (real attention, x_in / q_b / o_proj / router GEMVs, wait and MoE stand-ins); build only (needs
# an idle GPU and ~3-5 GB): $OUT/l2pf_bench (file header)
# shellcheck disable=SC2086
build l2pf_bench $NVCC -O3 -std=c++17 $GEN $K1H2_FLAGS ${GR8_NR:+-DG6_NR=$GR8_NR} -I "$REPO/engine/kernels" \
  -o "$OUT/l2pf_bench" "$REPO/engine/kernels/tests/l2pf_bench.cu" "$REPO/engine/attn/attn.cu" \
  "$REPO/engine/attn/bind.cu" "$REPO/engine/attn/ops_glm.cu" -lcublas "${KNOBS[@]}"
# GR8_ROUTE_OVL (the shared expert on a forked stream during the router) against the default MoE layer: byte-exact check
# + bench on a synthetic TP4 layer; build only (needs an idle GPU and ~2.5 GB): $OUT/k1_ovl_test (file header)
# shellcheck disable=SC2086
build k1_ovl_test $NVCC -O2 -std=c++17 $S2 -I "$REPO/engine/kernels" -o "$OUT/k1_ovl_test" \
  "$REPO/engine/kernels/tests/k1_ovl_test.cu" "${KNOBS[@]}"
# Steering projection/consume/finalizer fixtures (and S2 batched steering, s2_steer_test): compile in CI even without a
# GPU. No model or fabric service needed.
STEERING="s2_steer_test steering_consume_test steering_shared_test steering_boot_cuda_test steering_cuda_test"
for test in $STEERING; do
  # shellcheck disable=SC2086
  build $test $NVCC -O2 -std=c++17 $S2 -o "$OUT/$test" "$REPO/engine/runtime/tests/$test.cu" -lcublas "${KNOBS[@]}"
done
# dense trellis matrices, at each width with that width's engine flags, and in the Flash geometry at its widths: build
# only. dtr_test runs on the release's files (engine/ref/dtr_ref.py)
for n in $WIDTHS; do
  f="-DG6_NR=$n -DDEC_S2"
  case " ${F12_WIDTHS:-4} " in *" $n "*) f="$f -DPF_F12" ;; esac
  # shellcheck disable=SC2086
  build dtr_test_tp$n $NVCC -O2 -std=c++17 $GEN $K1H2_FLAGS $f -I "$REPO/engine/kernels" -o "$OUT/dtr_test_tp$n" \
    "$REPO/engine/kernels/tests/dtr_test.cu" -lcublas "${KNOBS[@]}"
done
for n in 2 3 4; do
  # shellcheck disable=SC2086
  build dtr_test_flash_tp$n $NVCC -O2 -std=c++17 $GEN ${K1H2_FLAGS/-DPF_FC2GL=6/-DPF_FC2GL=4} -DG6_MODEL_FLASH=1 \
    -DG6_NR=$n -DDEC_S2 -I "$REPO/engine/kernels" -o "$OUT/dtr_test_flash_tp$n" "$REPO/engine/kernels/tests/dtr_test.cu" \
    -lcublas "${KNOBS[@]}"
done
# TR3 / exlr8-e4m3 decode unit test; runs here when a GPU is visible, else later by hand: $OUT/tr3_e4m3_test
build tr3_e4m3_test $NVCC -O2 -std=c++17 $GEN -o "$OUT/tr3_e4m3_test" "$REPO/engine/kernels/tests/tr3_e4m3_test.cu" \
  -lcublas "${KNOBS[@]}"
# gpuprobe: the GPU part of the rank health probe (tools/health/rank_probe.py)
build gpuprobe $NVCC -O2 -std=c++17 $GEN -o "$OUT/gpuprobe" "$REPO/tools/health/gpuprobe.cu"
# the prefetch queues (GR8_L2PF targets, GR8_MTP_L2PF's draft eh_proj / head targets): order and capacity, CPU
# shellcheck disable=SC2086
build l2pf_queue_test nvcc -O2 -std=c++17 $GEN $K1H2_FLAGS ${GR8_NR:+-DG6_NR=$GR8_NR} -I "$REPO/engine/kernels" \
  -o "$OUT/l2pf_queue_test" "$REPO/engine/kernels/tests/l2pf_queue_test.cu"
# K1 split choice (GR8_K1_SPLIT table vs triple, the fused remap's KS1 x WG >= 2 guard): host only, runs everywhere
# shellcheck disable=SC2086
build k1_split_test nvcc -O2 -std=c++17 $S2 -I "$REPO/engine/kernels" -o "$OUT/k1_split_test" \
  "$REPO/engine/kernels/tests/k1_split_test.cu" "${KNOBS[@]}"
# fabric groups over real verbs (#154): built here, run by hand on the cluster (one process per box; see its header)
build fabric_group_test $NVCC -O2 -std=c++17 -I "$REPO/engine/comms" -o "$OUT/fabric_group_test" \
  "$REPO/engine/comms/tests/fabric_group_test.cpp" "$REPO/engine/comms/fabric.cpp" -libverbs "${KNOBS[@]}"
# flat reduce-scatter slices (Fabric::Rs::flat) over real verbs, mesh or ring: built here, run on the cluster (see its header)
build fabric_rs_flat_test $NVCC -O2 -std=c++17 -I "$REPO/engine/comms" -o "$OUT/fabric_rs_flat_test" \
  "$REPO/engine/comms/tests/fabric_rs_flat_test.cpp" "$REPO/engine/comms/fabric.cpp" -libverbs "${KNOBS[@]}"
# the adaptive draft length decision, CPU
build test_spec_len $GXX -O2 -std=c++17 -o "$OUT/test_spec_len" "$REPO/engine/tests/test_spec_len.cpp"
# the prefix-reuse state machine (--reuse-prefix and its grid rules), CPU
build test_prefix_reuse $GXX -O2 -std=c++17 -I "$REPO/engine/runtime" -o "$OUT/test_prefix_reuse" \
  "$REPO/engine/tests/test_prefix_reuse.cpp"
wait
cat "$TL"/*.log
if ls "$TL"/*.failed > /dev/null 2>&1; then echo "test builds failed: $(cd "$TL" && ls -- *.failed | sed 's/\.failed$//')"; exit 1; fi
check() {   # check <test>: run $OUT/<test>; print its last line, named, or all of it and fail
  local t
  t=$("$OUT/$1") || { echo "$t"; echo "$1 FAILED"; exit 1; }
  t=$(echo "$t" | tail -1)
  case "$t" in "$1"*) echo "$t" ;; *) echo "$1: $t" ;; esac
}
check test_spec_len
check test_prefix_reuse
check l2pf_queue_test
check k1_split_test
if nvidia-smi -L > /dev/null 2>&1; then
  check tr3_e4m3_test
  for test in $STEERING; do "$OUT/$test"; done
else
  echo "tr3_e4m3_test and the steering tests ($STEERING): compiled; no GPU visible, not run"
fi
echo "dtr_test_tp<N>, dtr_test_flash_tp<N>, l2pf_bench, k1_ovl_test: built (they need release files or an idle GPU)"
echo "fabric_group_test, fabric_rs_flat_test: built (cluster tests, not run here)"
[ -f "$GR8_NVCC_CACHE_LOG" ] && echo "nvcc cache (tests): $(grep -c '^hit' "$GR8_NVCC_CACHE_LOG") hit, $(grep -c '^miss' "$GR8_NVCC_CACHE_LOG") miss"
exit 0
