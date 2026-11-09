#!/bin/bash
# Build and run the Engram GPU test (needs a GB10 / sm_121a). Nothing is deleted.
#   run_gpu_test.sh <PREFIX.engram.cmap> <vectors.bin> <scratch dir>        (cmap + vectors: gen_engram.py, or run_hash_test.sh's scratch dir)
# The scratch dir should be on a real filesystem (sparse files, O_DIRECT); it holds, per rank, 2 sparse w8 files of 6.1 GB logical size, 2 sparse s8 files of 192 MB and a few MB of real rows. Device use is about 0.4 GB.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ENG="$HERE/../.."
CM="$1"; VEC="$2"; SCR="$3"; mkdir -p "$SCR"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
# section 7d links the attention code's tensor-core GEMM (ds41_gemm_mx_tc, ds41_attn.cu, built -O3 like the production attention TU); without that file, drop the .o and 7d reports "not linked"
"$NVCC" -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I"$ENG" -diag-suppress 20011,20014 \
    -c -o "$SCR/ds41_attn.o" "$ENG/attn/ds41_attn.cu"
"$NVCC" -std=c++17 -O2 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_DS41=1 -I"$ENG" -diag-suppress 20011,20014 \
    -Xcompiler -ffp-contract=off,-pthread -o "$SCR/test_engram_gpu" "$HERE/test_engram_gpu.cu" "$SCR/ds41_attn.o"
# rank 1 of 16: ordinary ids; rank 15 of 16: the highest ids (> 360 M) and file offsets above 4 GiB
RC=0; for R in 1 15; do "$SCR/test_engram_gpu" "$CM" "$VEC" "$SCR" "$R" || RC=1; done; exit $RC   # both ranks run even if the first fails
