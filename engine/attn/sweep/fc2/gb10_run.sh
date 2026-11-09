#!/usr/bin/env bash
# fc2 sweep bench on ONE GB10 (sm_121a). NOT run by the author: this box has no GB10 access for this task.
#
#   engine/attn/sweep/fc2/gb10_run.sh [VARIANT ...]          # default: off tmpl  (the served kernel and its private copy)
#   SUITE=all engine/attn/sweep/fc2/gb10_run.sh off myopt    # suite: repro (default, M=8192 served call) | all | smoke
#
# Run it ON the GB10 host, from a checkout that has the variant directories. One GPU, one process at a time:
# the measurement stage takes an exclusive flock so two bench runs never overlap.
# It never writes outside OUT, never deletes, never touches another box.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="$(cd "$HERE/.." && pwd)"
NVCC="${NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
SUITE="${SUITE:-repro}"
POOLS="${POOLS:-3}"       # rotating fixture pools (fresh inputs each launch window); weights alone are 601,620,480 B at 384 experts
WARMUP="${WARMUP:-3}"
REPS="${REPS:-5}"
ITERS="${ITERS:-5}"
LOCK="${LOCK:-/tmp/attn-sweep-gb10.lock}"
OUT="${OUT:-$HERE/runs/gb10-$(date -u +%Y%m%dT%H%M%SZ)-$SUITE}"
VARIANTS=("$@"); [ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(off tmpl)

[ -x "$NVCC" ] || { echo "nvcc not found: $NVCC (set NVCC=...)" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "output exists, pick a new one: $OUT" >&2; exit 2; }
for v in "${VARIANTS[@]}"; do
  [ -f "$HERE/variants/$v/variant.cuh" ] || { echo "no variant: $HERE/variants/$v/variant.cuh" >&2; exit 2; }
done
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free --format=csv

# Stage 1: build for sm_121a. No GPU needed, no lock held.
python3 "$SWEEP/run.py" --kernel "$HERE/fc2.json" --arch 121a --nvcc "$NVCC" --suite "$SUITE" \
  --variants "${VARIANTS[@]}" --build-only --out "$OUT"

# Stage 2: measure, serialized by the lock. --timeout 900: the M=8192 / 384-expert fixtures build large synthetic weight
# and routing sets on the host, so the default 180 s per case is too low.
exec 9>"$LOCK"
flock 9
python3 "$SWEEP/run.py" --run-only --out "$OUT" --pools "$POOLS" --warmup "$WARMUP" --reps "$REPS" --iters "$ITERS" \
  --timeout 900

echo "DONE. Tables: $OUT/summary.csv (and results.jsonl, run-manifest.json). Copy $OUT back for the report."
