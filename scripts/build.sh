#!/bin/bash
# Build the K1H2 engine (g6m4 = one rank on the real fabric, m4sim = one GPU simulating all NR ranks) on ONE Spark.
# Needs CUDA 13 (nvcc with sm_121a), g++ with C++20, libibverbs-dev, cuBLAS. A few minutes on a GB10, CPU only.
#   scripts/build.sh [OUT_DIR]            default OUT_DIR = ./bin
#   G6_NR=4 scripts/build.sh bin4         the TP4 engine (four Sparks, packs from build_packs.sh with the TP4 codes), with
#                                         DEC_S2 (the multi-session path launch_tp4.sh runs) and PF_F12 (the fat-CTA prefill fc1)
#   scripts/build.sh bin6                 the TP6 engine (G6_NR=6 is the default), with DEC_S2 (launch_tp6.sh runs it)
#   G6_NR=3 scripts/build.sh bin3         the TP3 engine, with DEC_S2 (launch_tp3.sh runs it)
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
# The S2 page allocator (engine/kvman-rs) is not optional: g6m4 always links target/release/libg6kvman.a, built the
# same way.
if command -v cargo > /dev/null 2>&1; then
  cargo build --release --quiet --manifest-path "$REPO/engine/loader-rs/Cargo.toml"
  cargo build --release --quiet --manifest-path "$REPO/engine/kvman-rs/Cargo.toml"
fi
G6LOAD_BIN=$REPO/engine/loader-rs/target/release/g6load
if [ -x "$G6LOAD_BIN" ]; then cp "$G6LOAD_BIN" "$OUT/"; fi
# the KV node ./gr8 kv up runs
KVNODE_BIN=$REPO/engine/kvman-rs/target/release/kvman-remote
if [ -x "$KVNODE_BIN" ]; then cp "$KVNODE_BIN" "$OUT/"; fi
cd "$REPO/engine/runtime"
S2_FLAG=""
case "${G6_NR:-6}" in
  4|3|6)
    [ "${PLAIN:-0}" = 1 ] && { echo "PLAIN=1 is retired: TP4, TP6 and TP3 builds always have the multi-session path (DEC_S2)"; exit 2; }
    S2_FLAG="-DDEC_S2"
    # PF_F12: F12_NE=2 must divide the expert blocks per rank (4 at TP4, 3 at TP6)
    [ "${G6_NR:-6}" = 4 ] && S2_FLAG="$S2_FLAG -DPF_F12" ;;
esac
EXTRA="$K1H2_FLAGS ${G6_NR:+-DG6_NR=$G6_NR} $S2_FLAG ${EXTRA:-}" OUT="$OUT" bash build_m4.sh "${ARCH:-121a}" all
if [ "${BUNDLE_LIBS:-0}" = 1 ]; then
  mkdir -p "$OUT/lib"
  cp -L /usr/local/cuda/lib64/libcublas.so.13 /usr/local/cuda/lib64/libcublasLt.so.13 "$OUT/lib/"
fi
strings "$OUT/g6m4" | grep -q "G6_CP_PF_HASH2 build" && echo "HASH2 build marker: present"
md5sum "$OUT/g6m4" "$OUT/m4sim"
export PATH=/usr/local/cuda/bin:$PATH
# the adaptive draft length decision, CPU
g++ -O2 -std=c++17 -o "$OUT/test_spec_len" "$REPO/engine/tests/test_spec_len.cpp"
t=$("$OUT/test_spec_len") || { echo "$t"; echo "test_spec_len FAILED"; exit 1; }
echo "test_spec_len: $(echo "$t" | tail -1)"
# the prefix-reuse state machine (--reuse-prefix and its grid rules), CPU
g++ -O2 -std=c++17 -I "$REPO/engine/runtime" -o "$OUT/test_prefix_reuse" "$REPO/engine/tests/test_prefix_reuse.cpp"
t=$("$OUT/test_prefix_reuse") || { echo "$t"; echo "test_prefix_reuse FAILED"; exit 1; }
echo "test_prefix_reuse: $(echo "$t" | tail -1)"
# gpuprobe: the GPU part of the rank health probe (tools/health/rank_probe.py); stage.sh copies it to every node
nvcc -O2 -std=c++17 -gencode=arch=compute_${ARCH:-121a},code=sm_${ARCH:-121a} -o "$OUT/gpuprobe" "$REPO/tools/health/gpuprobe.cu"
# TR3 / exlr8-e4m3 decode unit test; runs here when a GPU is visible, else later by hand: $OUT/tr3_e4m3_test
nvcc -O2 -std=c++17 -gencode=arch=compute_${ARCH:-121a},code=sm_${ARCH:-121a} -o "$OUT/tr3_e4m3_test" \
  "$REPO/engine/kernels/tests/tr3_e4m3_test.cu" -lcublas
# dense trellis matrices: build only. dtr_test runs on the release's files (engine/ref/dtr_ref.py)
nvcc -O2 -std=c++17 -gencode=arch=compute_${ARCH:-121a},code=sm_${ARCH:-121a} $K1H2_FLAGS ${G6_NR:+-DG6_NR=$G6_NR} $S2_FLAG \
  -I "$REPO/engine/kernels" -o "$OUT/dtr_test" "$REPO/engine/kernels/tests/dtr_test.cu" -lcublas
if nvidia-smi -L > /dev/null 2>&1; then
  t=$("$OUT/tr3_e4m3_test") || { echo "$t"; echo "tr3_e4m3_test FAILED"; exit 1; }
  echo "tr3_e4m3_test: $(echo "$t" | tail -1)"
else
  echo "tr3_e4m3_test: no GPU visible, not run"
fi
