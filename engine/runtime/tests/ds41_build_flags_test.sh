#!/bin/bash
# scripts/ds41_build_flags.sh:
#   GR8_DS_TRUNK_BF16 unset, empty or 1 -> -DDS_MHC_TRUNK_BF16, 0 -> omitted, anything else -> exit 2.
#   DS41_MX unset, empty or 48 -> -DG6_MX=48, 32|40 -> -DG6_MX=<N>, anything else -> exit 2.
set -eu
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
F="$REPO/scripts/ds41_build_flags.sh"
fail() { echo "ds41_build_flags_test: $*" >&2; exit 1; }
[ "$(env -u GR8_DS_TRUNK_BF16 -u DS41_MX bash "$F")" = "-DDS_MHC_TRUNK_BF16 -DG6_MX=48" ] || fail "defaults must print -DDS_MHC_TRUNK_BF16 -DG6_MX=48"
[ "$(GR8_DS_TRUNK_BF16= DS41_MX= bash "$F")" = "-DDS_MHC_TRUNK_BF16 -DG6_MX=48" ] || fail "empty must print -DDS_MHC_TRUNK_BF16 -DG6_MX=48"
[ "$(GR8_DS_TRUNK_BF16=0 env -u DS41_MX bash "$F")" = "-DG6_MX=48" ] || fail "trunk=0 must print -DG6_MX=48"
[ "$(GR8_DS_TRUNK_BF16=1 DS41_MX=48 bash "$F")" = "-DDS_MHC_TRUNK_BF16 -DG6_MX=48" ] || fail "trunk=1 mx=48 must print both"
[ "$(GR8_DS_TRUNK_BF16=1 DS41_MX=40 bash "$F")" = "-DDS_MHC_TRUNK_BF16 -DG6_MX=40" ] || fail "mx=40 must print -DG6_MX=40"
[ "$(GR8_DS_TRUNK_BF16=0 DS41_MX=32 bash "$F")" = "-DG6_MX=32" ] || fail "trunk=0 mx=32 must print -DG6_MX=32"
rc=0; GR8_DS_TRUNK_BF16=2 bash "$F" >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || fail "trunk=2 must exit 2 (got $rc)"
rc=0; DS41_MX=24 bash "$F" >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || fail "DS41_MX=24 must exit 2 (got $rc)"
echo "ds41_build_flags_test: ok"
