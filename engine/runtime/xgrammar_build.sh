#!/bin/bash
# Build XGrammar's C++ library (Python bindings off) from SRC into DIR/libxgrammar.a: build_m4.sh, and the image's
# xgrammar stage, which builds it once for every engine stage.
#   engine/runtime/xgrammar_build.sh SRC DIR      the log is DIR/build.log
set -eu
SRC=$1 XB=$2
mkdir -p "$XB"
echo 'set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)' > "$XB/config.cmake"   # the tree reads it from -B
{ cmake -S "$SRC" -B "$XB" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_CXX_FLAGS=-Wno-deprecated-declarations -DCMAKE_POSITION_INDEPENDENT_CODE=ON &&
  cmake --build "$XB" -j "$(nproc)" --target xgrammar; } > "$XB/build.log" 2>&1 || { echo "XGrammar: $XB/build.log"; exit 1; }
