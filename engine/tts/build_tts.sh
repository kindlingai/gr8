#!/bin/bash
# Build g6tts (native CUDA TP=1 Drama 3 TTS engine) and g6tts_bench.
# Usage:
#   engine/tts/build_tts.sh [OUT_DIR]
#   ARCH=121a engine/tts/build_tts.sh bin
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
  echo "build_tts.sh: missing $GL (run: cargo build --release --locked --manifest-path engine/loader-rs/Cargo.toml)" >&2
  exit 1
fi

NVCC=${NVCC:-nvcc}
FLAGS=(-O3 -std=c++17 -lineinfo -gencode="arch=compute_${ARCH},code=sm_${ARCH}"
       -I "$REPO/engine/tts" -I "$REPO/engine/loader-rs" -I "$REPO/engine/runtime")
LIBS=("$GL" -lcublas -lpthread -ldl -lm)

echo "build_tts.sh: compiling g6tts and g6tts_bench for sm_${ARCH} -> $OUT"
"$NVCC" "${FLAGS[@]}" "$REPO/engine/tts/g6tts.cu" "${LIBS[@]}" -o "$OUT/g6tts" &
P1=$!
"$NVCC" "${FLAGS[@]}" "$REPO/engine/tts/g6tts_bench.cu" "${LIBS[@]}" -o "$OUT/g6tts_bench" &
P2=$!
wait "$P1"
wait "$P2"
echo "build_tts.sh: built $OUT/g6tts and $OUT/g6tts_bench"
