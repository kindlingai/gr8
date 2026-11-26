#!/bin/bash
# The extra nvcc -D flags of the DS41 engine build (scripts/build.sh MODEL=ds41) from the build knobs; prints them on one line.
#   GR8_DS_TRUNK_BF16     1 (the default) -> -DDS_MHC_TRUNK_BF16: the mHC trunk (Model::resid4, ds_mhc::Buffers::streams) is stored as BF16, not FP32.
#                         The trunk only holds BF16-valued numbers, so no bit of any result changes (engine/mhc/ds_mhc.cuh); it halves
#                         the trunk's DRAM traffic in the prefill mHC kernels and the trunk's memory.
#   DS41_MX               48 (the default), 40 or 32 -> -DG6_MX=<N>: decode rows per step (engine/g6mx.h). 48 holds 8 sessions x 6
#                         DSpark verify rows in one step, matching the published image and scripts/engine_args.sh.
# Any other value of either variable: a message on stderr, exit 2.
set -eu
flags=()
case "${GR8_DS_TRUNK_BF16:-1}" in
  0) ;;
  1) flags+=("-DDS_MHC_TRUNK_BF16") ;;
  *) echo "GR8_DS_TRUNK_BF16=$GR8_DS_TRUNK_BF16: 0 or 1" >&2; exit 2 ;;
esac
case "${DS41_MX:-48}" in
  32|40|48) flags+=("-DG6_MX=${DS41_MX:-48}") ;;
  *) echo "DS41_MX=$DS41_MX: 32, 40 or 48" >&2; exit 2 ;;
esac
echo "${flags[*]}"
