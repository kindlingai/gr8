#!/usr/bin/env bash
# CPU-only gate for the DS shifted-HC trunk: stage split == DeepGEMM mhc_reference, then ds_cpu.h == torch reference.
# usage: ds_mhc_cpu_test.sh [work_dir]   (needs g++, python3 + torch CPU; no GPU)
# DS_MHC_VENDOR_TEST: path to DeepGEMM's tests/test_mega_mhc.py; the stage-split check is skipped without it.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
W="${1:-/tmp/ds_mhc_cpu_test}"; mkdir -p "$W"
export CUDA_VISIBLE_DEVICES=
VENDOR="${DS_MHC_VENDOR_TEST:-}"
if [ -n "$VENDOR" ] && [ -f "$VENDOR" ]; then python3 "$REPO/engine/mhc/ds_ref.py" check "$VENDOR"; else echo "SKIP vendor check: DS_MHC_VENDOR_TEST unset or missing"; fi
g++ -O2 -std=c++17 -I "$REPO/engine" -o "$W/ds_mhc_cpu_test" "$REPO/engine/runtime/tests/ds_mhc_cpu_test.cpp"
rc=0
for cfg in "3 256 6 50 1e-20 1" "2 5120 4 1000 1e-20 2" "5 512 8 64 1e-6 3"; do
  python3 "$REPO/engine/mhc/ds_ref.py" gen "$W/case.bin" $cfg
  "$W/ds_mhc_cpu_test" "$W/case.bin" | grep -E "ds_mhc_cpu_test:|FAIL|PASS" || rc=1
done
exit $rc
