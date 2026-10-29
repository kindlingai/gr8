#!/bin/bash
# build the M4 drivers. usage: build_m4.sh <arch: 121a (DGX Spark GB10) | 120a (RTX 50xx)> [sim|g6|all|dfl|rs]
# sim / g6 / all: m4sim and g6m4, one binary each that runs every TP width in WIDTHS (default 6; below).
# JOBS: compiles at once (default: the CPU count; each takes under 1 GB).
# m4sim: one GPU, NR simulated ranks. g6m4: one rank on the real fabric (needs libibverbs; Spark only).
# dfl: dfltest and dflreplay only, the DFlash2 drafter step of all ranks on one GPU (engine/ref/dflash2_check.py; the
#      G6_DFL_BODY_I4 replay of a G6_DFL_DUMP)
# rs: rstest only, the head kernels of all ranks on one GPU: rejection sampling vs the target distribution.
# dfl and rs build the one width EXTRA names (EXTRA="-DG6_NR=4 -DDEC_S2"; default 6).
# EXTRA=-DG6_MODEL_FLASH=1: the GLM 5.3 Flash geometry (engine/model_geom.h); scripts/build.sh MODEL=flash names the results *_flash.
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
# every compile goes through nvcc_cache.sh (#32): a rebuild compiles only what an edit reaches. G6_NVCC_CACHE=0: plain nvcc
if [ "${G6_NVCC_CACHE:-1}" = 0 ]; then NVCC=nvcc; GXX=g++; else NVCC="./nvcc_cache.sh nvcc"; GXX="./nvcc_cache.sh g++"; fi
export G6_NVCC_CACHE_LOG=$O/nvcc_cache.calls
rm -f $G6_NVCC_CACHE_LOG
cache_counts() { [ -f $G6_NVCC_CACHE_LOG ] && echo "nvcc cache: $(grep -c '^hit' $G6_NVCC_CACHE_LOG) hit, $(grep -c '^miss' $G6_NVCC_CACHE_LOG) miss, $(grep -c '^pass' $G6_NVCC_CACHE_LOG) not cacheable"; return 0; }
# the knob table's C API (engine/knobs-rs, gr8_knobs.h): every binary that reads a knob links it. scripts/build.sh
# builds it with cargo
KNOBLIB=../knobs-rs/target/release/libgr8knobs.a
[ -f "$KNOBLIB" ] || { echo "need $KNOBLIB: cargo build --release --features capi --manifest-path engine/knobs-rs/Cargo.toml"; exit 1; }
KNOBS="$KNOBLIB -lpthread -ldl -lm"
# the attention TU, its binder and its op table (attn/ops.h): attn.cu + bind.cu + ops_glm.cu (GLM), flash_attn.cu +
# flash_bind.cu + ops_flash.cu (Flash: NoPE MLA + kpool indexer)
ATTN_CU=../attn/attn.cu; ATTN_BIND_CU=../attn/bind.cu; ATTN_OPS_CU=../attn/ops_glm.cu
case " $EXTRA " in *" -DG6_MODEL_FLASH=1 "*) ATTN_CU=../attn/flash_attn.cu; ATTN_BIND_CU=../attn/flash_bind.cu; ATTN_OPS_CU=../attn/ops_flash.cu ;; esac
if [ "$WHAT" = rs ]; then
  nice -n 19 $NVCC -O2 -std=c++17 $GEN -o $OUT/rstest rstest.cu $KNOBS
  ls -la $OUT/rstest
  cache_counts
  exit 0
fi
nice -n 19 $GXX -O3 -std=c++20 $CPU -pthread -c ../loader/g6pack.cpp -o $O/g6pack.o
# Flash: the kvchunks regions of G6_FLASH_KV_CHUNKS (runtime/flash_kv.h), one object for every width
KVC=
if [ "$ATTN_CU" = ../attn/flash_attn.cu ]; then
  nice -n 19 $NVCC -O2 -std=c++17 $GEN -c ../attn/kvchunks.cu -o $O/kvchunks.o; KVC="$O/kvchunks.o"
fi
nice -n 19 $GXX -O3 -std=c++20 $CPU -pthread -I/usr/local/cuda/include -c ../loader/devimage.cpp -o $O/devimage.o
if [ "$WHAT" = dfl ]; then
  nice -n 19 $NVCC -O3 -std=c++17 $GEN -c $ATTN_CU -o $O/attn.o
  nice -n 19 $NVCC -O3 -std=c++17 $GEN -c $ATTN_BIND_CU -o $O/attn_bind.o
  nice -n 19 $NVCC -O3 -std=c++17 $GEN -c $ATTN_OPS_CU -o $O/attn_ops.o
  nice -n 19 $NVCC -O2 -std=c++17 $GEN -o $OUT/dfltest dfltest.cu $O/attn.o $O/attn_bind.o $O/attn_ops.o $KVC $O/g6pack.o $O/devimage.o -lcublas $KNOBS
  nice -n 19 $NVCC -O2 -std=c++17 $GEN -o $OUT/dflreplay dflreplay.cu $O/attn.o $O/attn_bind.o $O/attn_ops.o $KVC $O/g6pack.o $O/devimage.o -lcublas $KNOBS
  ls -la $OUT/dfltest $OUT/dflreplay
  cache_counts
  exit 0
fi
# the Rust pack loader (engine/loader-rs, G6_RS_LOADER) when its library is built. scripts/build.sh builds it with cargo
RSLIB=../loader-rs/target/release/libg6load.a
RSDEF=""
RSLINK=""
if [ -f "$RSLIB" ]; then RSDEF="-DG6_RS_LOADER"; RSLINK="$RSLIB -ldl -lm"; fi
# request grammars (grammar.h): G6_XGRAMMAR=DIR, an XGrammar source tree (scripts/build.sh finds the one the front end's
# xgrammar crate pins). Its library is built once per object directory (xgrammar_build.sh), or comes prebuilt from
# G6_XGRAMMAR_LIB. g6m4 then takes --grammar-dir; without G6_XGRAMMAR a gram request is refused
GRDEF=""
GRLINK=""
if [ -n "${G6_XGRAMMAR:-}" ] && [ "$WHAT" != sim ]; then
  XLIB=${G6_XGRAMMAR_LIB:-$O/xgrammar/libxgrammar.a}
  if [ ! -f "$XLIB" ]; then
    [ -z "${G6_XGRAMMAR_LIB:-}" ] || { echo "G6_XGRAMMAR_LIB=$G6_XGRAMMAR_LIB: no such file"; exit 1; }
    bash xgrammar_build.sh "$G6_XGRAMMAR" "$O/xgrammar" || exit 1
  fi
  GRDEF="-DG6_GRAMMAR=1"
  g++ -O2 -std=c++17 -DG6_GRAMMAR=1 -I"$G6_XGRAMMAR/include" -I"$G6_XGRAMMAR/3rdparty/dlpack/include" \
      -I"$G6_XGRAMMAR/3rdparty/picojson" -c grammar.cc -o $O/grammar.o || exit 1
  GRLINK="$O/grammar.o $XLIB"
  g++ -O2 -std=c++17 -DG6_GRAMMAR=1 -I. -o $OUT/grammar_test tests/grammar_test.cc $GRLINK -lpthread || exit 1
  $OUT/grammar_test || exit 1
fi
# the S2 page allocator (engine/kvman-rs, wrapped by s2/pool.h): g6m4 and m4sim link it. scripts/build.sh builds it with cargo
KVLIB=../kvman-rs/target/release/libg6kvman.a
[ -f "$KVLIB" ] || { echo "g6m4 and m4sim need $KVLIB: cargo build --release --manifest-path engine/kvman-rs/Cargo.toml"; exit 1; }

# One binary for every TP width. Each width compiles the sources that depend on it (m4sim.cu / g6m4.cu and the
# attn.cu / bind.cu they link) with that width's flags, as a one-width build did, so its kernels are the same ones.
# ld -r joins them into the width's instance; objcopy renames its main g6w<N>_main and makes every other symbol local,
# since the widths define the same names with different code and a global name would be merged across them.
# --force-group-allocation settles the instance's inline functions inside it: the final link would otherwise keep one
# width's copy of each. -fno-gnu-unique (host code only) binds inline variables and the static locals of inline
# functions weak instead of unique, which objcopy cannot make local. Symbols the shared objects (fabric.o, the pack
# loaders, the page allocator) define as well stay global and weak, so the process has one copy of each (fabric.o's
# progress CPU lists). widths.cpp is the main that picks the instance.
WIDTHS=${WIDTHS:-6}
# the flags of width N: the multi-session path (DEC_S2, which g6m4 needs) at every width, and PF_F12 (F12_NE=2 must
# divide the expert blocks per rank: 4 at TP4, 3 at TP6) at 4, except in the Flash build, whose fp16 MoE refuses it
wflags() {
  local f="-DG6_NR=$1 -DDEC_S2"
  case " ${F12_WIDTHS:-4} " in *" $1 "*) [ "$ATTN_CU" = ../attn/attn.cu ] && f="$f -DPF_F12" ;; esac   # PF_F12 (fat-CTA prefill fc1): TP4 by default; F12_WIDTHS="4 5" adds TP5; never in the Flash build
  echo "$f"
}
for n in $WIDTHS; do
  case $n in 1|2|3|4|5|6|8) ;; *) echo "WIDTHS: $n is not a TP width (1, 2, 3, 4, 5, 6 or 8)"; exit 2 ;; esac
done
J=${JOBS:-$(nproc)}
rm -f $O/*.log $O/*.failed
job() {   # job <log> <command..>: in the background, at most J at once; a failure leaves <log>.failed
  while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n || true; done
  { nice -n 19 "${@:2}" > "$1" 2>&1 || touch "$1.failed"; } &
}
if [ "$WHAT" != sim ]; then job $O/fabric.log $NVCC -O2 -std=c++17 $GEN -c ../comms/fabric.cpp -o $O/fabric.o; fi
for n in $WIDTHS; do
  F="$GEN $(wflags $n) -Xcompiler -fno-gnu-unique"
  job $O/attn.w$n.log $NVCC -O3 -std=c++17 $F -c $ATTN_CU -o $O/attn.w$n.o
  job $O/attn_bind.w$n.log $NVCC -O3 -std=c++17 $F -c $ATTN_BIND_CU -o $O/attn_bind.w$n.o
  job $O/attn_ops.w$n.log $NVCC -O3 -std=c++17 $F -c $ATTN_OPS_CU -o $O/attn_ops.w$n.o
  if [ "$WHAT" != g6 ]; then job $O/m4sim.w$n.log $NVCC -O2 -std=c++17 $F $RSDEF -c m4sim.cu -o $O/m4sim.w$n.o; fi
  if [ "$WHAT" != sim ]; then job $O/g6m4.w$n.log $NVCC -O2 -std=c++17 $F $RSDEF $GRDEF -c g6m4.cu -o $O/g6m4.w$n.o; fi
done
wait
cat $O/*.log
if ls $O/*.failed > /dev/null 2>&1; then echo "build_m4.sh failed: $(cd $O && ls *.failed | sed 's/\.log\.failed//')"; exit 1; fi
SHARED="$O/g6pack.o $O/devimage.o ${KVC%% *} $KVLIB $KNOBLIB"
if [ "$WHAT" != sim ]; then SHARED="$SHARED $O/fabric.o"; fi
if [ -n "$RSDEF" ]; then SHARED="$SHARED $RSLIB"; fi
nm --defined-only -g $SHARED 2> /dev/null | awk 'NF == 3 { print $3 }' | sort -u > $O/shared.syms
inst() {   # inst <bin> <width>: $O/<bin>.w<width>.i.o, the width's instance of <bin>
  local b=$O/$1.w$2
  ld -r --force-group-allocation -o $b.r.o $b.o $O/attn.w$2.o $O/attn_bind.w$2.o $O/attn_ops.w$2.o
  objcopy --redefine-sym main=g6w$2_main $b.r.o
  nm --defined-only -g $b.r.o | awk 'NF == 3 { print $3 }' | sort -u | comm -12 - $O/shared.syms > $b.weak
  { echo g6w$2_main; cat $b.weak; } > $b.keep
  objcopy --keep-global-symbols=$b.keep --weaken-symbols=$b.weak $b.r.o $b.i.o
}
if [ "$WHAT" != g6 ]; then
  I=""
  for n in $WIDTHS; do inst m4sim $n; I="$I $O/m4sim.w$n.i.o"; done
  nice -n 19 $NVCC -O2 -std=c++17 $GEN -DM4SIM -o $OUT/m4sim widths.cpp $I $O/g6pack.o $O/devimage.o $KVC $RSLINK $KVLIB $KNOBS -lcublas -lcublasLt
  ls -la $OUT/m4sim
fi
if [ "$WHAT" != sim ]; then
  I=""
  for n in $WIDTHS; do inst g6m4 $n; I="$I $O/g6m4.w$n.i.o"; done
  nice -n 19 $NVCC -O2 -std=c++17 $GEN $GRDEF -o $OUT/g6m4 widths.cpp $I $O/fabric.o $O/g6pack.o $O/devimage.o $KVC $GRLINK $RSLINK $KVLIB $KNOBS -lcublas -lcublasLt -libverbs
  if [ -n "$GRDEF" ]; then echo "g6m4: request grammars (XGrammar) linked"; fi
  if [ -n "$RSDEF" ]; then echo "g6m4: Rust pack loader linked"; else echo "g6m4: no $RSLIB, C++ pack loader only"; fi
  ls -la $OUT/g6m4
fi
cache_counts
echo "TP widths: $WIDTHS"