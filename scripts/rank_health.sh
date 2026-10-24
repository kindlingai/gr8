#!/bin/bash
# Rank health probe and boot admission for every rank (run on the head, with NO model booted on any box).
#   scripts/rank_health.sh [OUT_JSON] [OPTIONS]   default OUT_JSON = $G6_HOME/out/rank_health.json
# OPTIONS are ./gr8's (KNOBS.md): --peers, --ips, --ips2, --proxy-cpu, --g6-home, @FILE. The probe reaches the boxes
# over ssh by their PEERS names and needs IPS and IPS2 set (it does not discover them).
# Exit 0 = every rank admitted; 3 = refused (one line per refusal with a named reason); 2 = setup problem.
# The JSON is what `bench/gate/gate.py run --health` records. PROBE_ARGS="--min-clock-mhz 1800 ..." sets thresholds
# (tools/health/README.md).
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=""
[ $# -gt 0 ] && [ "${1#-}" = "$1" ] && [ "${1#@}" = "$1" ] && { OUT=$1; shift; }
get() { python3 "$REPO/tools/launch/launch.py" get "$1" "$@"; }
PEERS=$(get PEERS "$@") || { echo "rank_health: set PEERS (--peers)" >&2; exit 2; }
IPS=$(get IPS "$@") || { echo "rank_health: set IPS (--ips)" >&2; exit 2; }
IPS2=$(get IPS2 "$@") || { echo "rank_health: set IPS2 (--ips2)" >&2; exit 2; }
PROXY_CPU=$(get PROXY_CPU "$@")
G6_HOME=$(get G6_HOME "$@")
OUT=${OUT:-$G6_HOME/out/rank_health.json}
mkdir -p "$(dirname "$OUT")"
# shellcheck disable=SC2086
exec python3 "$REPO/tools/health/rank_probe.py" cluster --hosts "$PEERS" --ips "$IPS" --ips2 "$IPS2" \
  --home "$G6_HOME" --proxy-cpu "$PROXY_CPU" --out "$OUT" ${PROBE_ARGS:-}
