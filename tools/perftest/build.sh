#!/bin/bash
# tools/perftest/build.sh [OUT_DIR]: OUT_DIR/nccl_probe (default bin/) and the NCCL library it loads, OUT_DIR/lib/libnccl.so.2,
# copied from the build's own NCCL so the boxes need none installed. Needs nvcc and nccl.h / libnccl (the CUDA build image).
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$REPO/bin}
mkdir -p "$OUT/lib"
nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -Xlinker -rpath -Xlinker '$ORIGIN/lib' \
  "$REPO/tools/perftest/nccl_probe.cu" -lnccl -o "$OUT/nccl_probe"
lib=$(ldconfig -p | awk '/libnccl\.so\.2 /{print $NF; exit}')
[ -n "$lib" ] || { echo "build.sh: no libnccl.so.2 to copy" >&2; exit 1; }
cp -L "$lib" "$OUT/lib/libnccl.so.2"
echo "$OUT/nccl_probe, $OUT/lib/libnccl.so.2 ($(basename "$(readlink -f "$lib")"))"
