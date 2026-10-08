#!/bin/bash
# Build the K1H2 engine (g6m4 = one rank on the real fabric, m4sim = one GPU simulating all NR ranks) on ONE Spark.
# Needs CUDA 13 (nvcc with sm_121a), g++ with C++20, libibverbs-dev, cuBLAS. A few minutes on a GB10, CPU only.
#   scripts/build.sh [OUT_DIR]            default OUT_DIR = ./bin
#   G6_NR=4 scripts/build.sh bin4         the TP4 engine (four Sparks, packs from build_packs.sh with the TP4 codes)
#   BUNDLE_LIBS=1                         also copy cuBLAS (libcublas, libcublasLt) into OUT_DIR/lib, for nodes
#                                         without the CUDA toolkit (stage.sh copies it, launch_tp4.sh uses it)
# These -D flags are the K1H2 build; every one of them is an engine path that our A/B tests promoted
# (see docs/SETTINGS.md). Dropping one builds an older, slower or less-checked path, not a broken one.
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$REPO/bin}" && cd "${1:-$REPO/bin}" && pwd)
K1H2_FLAGS="-DPF_FC2GL=6 -DPF_GEMMF=1 -DPF_XQ8C=1 -DPF_RNWIDE=1 -DPF_GEMMR=1 -DPF_DRAFTKV -DPF_GEMMRF=1 -DPF_QOBH=1 \
-DDEC_SLOT=2 -DDEC_CPB=1 -DDEC_AVP=1 -DPF_QOROPE=1 -DPF_ISGRT=1 -DPF_ROPE2=1 -DDEC_CPB2=64 -DDEC_I4PAIR=1 -DSEL32 \
-DG6_CP=1 -DG6_CP_PF_HASH2=1 -DDEC_IDX31=1"
cd "$REPO/engine/runtime"
EXTRA="$K1H2_FLAGS ${G6_NR:+-DG6_NR=$G6_NR} ${EXTRA:-}" OUT="$OUT" bash build_m4.sh "${ARCH:-121a}" all
if [ "${BUNDLE_LIBS:-0}" = 1 ]; then
  mkdir -p "$OUT/lib"
  cp -L /usr/local/cuda/lib64/libcublas.so.13 /usr/local/cuda/lib64/libcublasLt.so.13 "$OUT/lib/"
fi
strings "$OUT/g6m4" | grep -q "G6_CP_PF_HASH2 build" && echo "HASH2 build marker: present"
md5sum "$OUT/g6m4" "$OUT/m4sim"
export PATH=/usr/local/cuda/bin:$PATH
# the adaptive draft length decision (#12), CPU
g++ -O2 -std=c++17 -o "$OUT/test_spec_len" "$REPO/engine/tests/test_spec_len.cpp"
t=$("$OUT/test_spec_len") || { echo "$t"; echo "test_spec_len FAILED"; exit 1; }
echo "test_spec_len: $(echo "$t" | tail -1)"
# gpuprobe: the GPU part of the rank health probe (tools/health/rank_probe.py); stage.sh copies it to every node
nvcc -O2 -std=c++17 -gencode=arch=compute_${ARCH:-121a},code=sm_${ARCH:-121a} -o "$OUT/gpuprobe" "$REPO/tools/health/gpuprobe.cu"
# TR3 / exlr8-e4m3 decode unit test (#19); runs here when a GPU is visible, else later by hand: $OUT/tr3_e4m3_test
nvcc -O2 -std=c++17 -gencode=arch=compute_${ARCH:-121a},code=sm_${ARCH:-121a} -o "$OUT/tr3_e4m3_test" \
  "$REPO/engine/kernels/tests/tr3_e4m3_test.cu" -lcublas
if nvidia-smi -L > /dev/null 2>&1; then
  t=$("$OUT/tr3_e4m3_test") || { echo "$t"; echo "tr3_e4m3_test FAILED"; exit 1; }
  echo "tr3_e4m3_test: $(echo "$t" | tail -1)"
else
  echo "tr3_e4m3_test: no GPU visible, not run"
fi
