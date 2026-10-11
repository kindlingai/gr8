#!/bin/bash
# build the M4 drivers. usage: build_m4.sh <arch: 121a (DGX Spark GB10) | 120a (RTX 50xx)> [sim|g6|all|dfl|rs|tests]
# tests: only the unit tests (tests/row_budget_test.cu, tests/s2_attn_test.cu) into $OUT; scripts/c1_tests.sh builds and runs them.
# m4sim: one GPU, six simulated ranks. g6m4: one rank on the real fabric (needs libibverbs; Spark only).
# dfl: dfltest only, the DFlash2 drafter step of all ranks on one GPU (engine/ref/dflash2_check.py).
# rs: rstest only, the head kernels of all ranks on one GPU: rejection sampling vs the target distribution.
set -e
cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH
A=${1:-121a}
WHAT=${2:-all}
GEN="-gencode=arch=compute_${A},code=sm_${A} ${EXTRA:-}"   # EXTRA: optional nvcc flags, e.g. EXTRA=-DPF_FC2GL=6
CPU=""
[ "$(uname -m)" = aarch64 ] && CPU="-mcpu=neoverse-v2+crypto"
OUT=${OUT:-../../bin}
O=$OUT/obj                 # one object dir per variant (OUT): concurrent flag / NR builds never share objects
mkdir -p $OUT $O
if [ "$WHAT" = tests ]; then
  nice -n 19 nvcc -O2 -std=c++17 $GEN -o $OUT/row_budget_test tests/row_budget_test.cu
  nice -n 19 nvcc -O2 -std=c++17 $GEN -o $OUT/s2_attn_test tests/s2_attn_test.cu ../attn/attn.cu ../attn/bind.cu -lcublas
  ls -la $OUT/row_budget_test $OUT/s2_attn_test
  exit 0
fi
if [ "$WHAT" = rs ]; then
  nice -n 19 nvcc -O2 -std=c++17 $GEN -o $OUT/rstest rstest.cu
  ls -la $OUT/rstest
  exit 0
fi
nice -n 19 g++ -O3 -std=c++20 $CPU -pthread -c ../loader/g6pack.cpp -o $O/g6pack.o
nice -n 19 g++ -O3 -std=c++20 $CPU -pthread -I/usr/local/cuda/include -c ../loader/devimage.cpp -o $O/devimage.o
nice -n 19 nvcc -O3 -std=c++17 $GEN -c ../attn/attn.cu -o $O/attn.o
nice -n 19 nvcc -O3 -std=c++17 $GEN -c ../attn/bind.cu -o $O/attn_bind.o
if [ "$WHAT" = dfl ]; then
  nice -n 19 nvcc -O2 -std=c++17 $GEN -o $OUT/dfltest dfltest.cu $O/attn.o $O/attn_bind.o $O/g6pack.o $O/devimage.o -lcublas -lpthread
  ls -la $OUT/dfltest
  exit 0
fi
# the Rust pack loader (engine/loader-rs, G6_RS_LOADER) when its library is built. scripts/build.sh builds it with cargo
RSLIB=../loader-rs/target/release/libg6load.a
RS=""
[ -f "$RSLIB" ] && RS="-DG6_RS_LOADER $RSLIB -ldl -lm"
# the S2 page allocator (engine/kvman-rs, wrapped by s2/pool.h): g6m4 and m4sim link it. scripts/build.sh builds it with cargo
KVLIB=../kvman-rs/target/release/libg6kvman.a
[ -f "$KVLIB" ] || { echo "g6m4 and m4sim need $KVLIB: cargo build --release --manifest-path engine/kvman-rs/Cargo.toml"; exit 1; }
if [ "$WHAT" != g6 ]; then
  nice -n 19 nvcc -O2 -std=c++17 $GEN -o $OUT/m4sim m4sim.cu $O/attn.o $O/attn_bind.o $O/g6pack.o $O/devimage.o $RS $KVLIB -ldl -lm -lcublas -lpthread
  ls -la $OUT/m4sim
fi
if [ "$WHAT" != sim ]; then
  nice -n 19 nvcc -O2 -std=c++17 $GEN -o $OUT/g6m4 g6m4.cu ../comms/fabric.cpp $O/attn.o $O/attn_bind.o $O/g6pack.o $O/devimage.o $RS $KVLIB -ldl -lm -lcublas -libverbs -lpthread
  [ -n "$RS" ] && echo "g6m4: Rust pack loader linked" || echo "g6m4: no $RSLIB, C++ pack loader only"
  ls -la $OUT/g6m4
fi
