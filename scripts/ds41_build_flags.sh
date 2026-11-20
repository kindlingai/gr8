#!/bin/bash
# The extra nvcc -D flags of the DS41 engine build (scripts/build.sh MODEL=ds41) from the build knobs; prints them on one line.
#   GR8_DS_TRUNK_BF16     1 (the default) -> -DDS_MHC_TRUNK_BF16: the mHC trunk (Model::resid4, ds_mhc::Buffers::streams) is stored as BF16, not FP32.
#                         The trunk only holds BF16-valued numbers, so no bit of any result changes (engine/mhc/ds_mhc.cuh); it halves
#                         the trunk's DRAM traffic in the prefill mHC kernels and the trunk's memory.
# Unset, empty or 1 prints -DDS_MHC_TRUNK_BF16. 0 prints nothing (the FP32 trunk). Any other value: a message on stderr, exit 2.
set -eu
case "${GR8_DS_TRUNK_BF16:-1}" in
  0) ;;
  1) echo "-DDS_MHC_TRUNK_BF16" ;;
  *) echo "GR8_DS_TRUNK_BF16=$GR8_DS_TRUNK_BF16: 0 or 1" >&2; exit 2 ;;
esac
