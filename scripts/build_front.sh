#!/bin/bash
# Build the Rust front end (frontend/g6front-rs, issue #15) on the head node and copy it to OUT_DIR.
#   scripts/build_front.sh [OUT_DIR]      default OUT_DIR = ./bin; launch_tp6.sh / launch_tp4.sh run it with FRONT=rs
# Needs rustup (the crate pins its compiler in rust-toolchain.toml; rustup fetches it on first build) and a C compiler
# (mimalloc and oniguruma are built from source). A few minutes, CPU only.
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$REPO/bin}" && cd "${1:-$REPO/bin}" && pwd)
cd "$REPO/frontend/g6front-rs"
cargo build --release --locked
cp target/release/g6front-rs "$OUT/"
md5sum "$OUT/g6front-rs" 2>/dev/null || md5 "$OUT/g6front-rs"
