#!/bin/bash
# scripts/ds41_build_flags.sh: GR8_DS_TRUNK_BF16 unset, empty or 0 -> no flags, 1 -> -DDS_MHC_TRUNK_BF16, anything else -> exit 2.
set -eu
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
F="$REPO/scripts/ds41_build_flags.sh"
fail() { echo "ds41_build_flags_test: $*" >&2; exit 1; }
[ -z "$(env -u GR8_DS_TRUNK_BF16 bash "$F")" ] || fail "unset must print nothing"
[ -z "$(GR8_DS_TRUNK_BF16= bash "$F")" ] || fail "empty must print nothing"
[ -z "$(GR8_DS_TRUNK_BF16=0 bash "$F")" ] || fail "0 must print nothing"
[ "$(GR8_DS_TRUNK_BF16=1 bash "$F")" = "-DDS_MHC_TRUNK_BF16" ] || fail "1 must print -DDS_MHC_TRUNK_BF16"
rc=0; GR8_DS_TRUNK_BF16=2 bash "$F" >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || fail "2 must exit 2 (got $rc)"
echo "ds41_build_flags_test: ok"
