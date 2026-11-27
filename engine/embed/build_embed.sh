#!/bin/bash
# Build g6embed (native CUDA TP=1 dense embedding engine) and g6embed_bench.
# Usage:
#   engine/embed/build_embed.sh [OUT_DIR]
#   ARCH=121a engine/embed/build_embed.sh bin
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(mkdir -p "${1:-$REPO/bin}" && cd "${1:-$REPO/bin}" && pwd)

if [ -z "${ARCH:-}" ]; then
  case "$(uname -m)" in
    x86_64|amd64) ARCH=120a ;;
    *)            ARCH=121a ;;
  esac
fi

if command -v cargo > /dev/null 2>&1; then
  cargo build --release --locked --quiet --manifest-path "$REPO/engine/loader-rs/Cargo.toml"
fi

GL="$REPO/engine/loader-rs/target/release/libg6load.a"
if [ ! -f "$GL" ]; then
  echo "build_embed.sh: missing $GL (run: cargo build --release --locked --manifest-path engine/loader-rs/Cargo.toml)" >&2
  exit 1
fi

NVCC=${NVCC:-nvcc}
FLAGS=(-O3 -std=c++17 -lineinfo -gencode="arch=compute_${ARCH},code=sm_${ARCH}"
       -I "$REPO/engine/embed" -I "$REPO/engine/loader-rs")
LIBS=("$GL" -lcublas -lpthread -ldl -lm)

echo "build_embed.sh: compiling g6embed and g6embed_bench for sm_${ARCH} -> $OUT"
"$NVCC" "${FLAGS[@]}" "$REPO/engine/embed/g6embed.cu" "${LIBS[@]}" -o "$OUT/g6embed" &
P1=$!
"$NVCC" "${FLAGS[@]}" "$REPO/engine/embed/g6embed_bench.cu" "${LIBS[@]}" -o "$OUT/g6embed_bench" &
P2=$!
wait "$P1"
wait "$P2"
echo "build_embed.sh: built $OUT/g6embed and $OUT/g6embed_bench"
