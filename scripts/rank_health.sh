#!/bin/bash
# Rank health probe + boot admission for the six ranks (run on the head, with NO model booted on any node).
#   scripts/rank_health.sh [OUT_JSON]       default OUT_JSON = $G6_HOME/out/rank_health.json
# Exit 0 = every rank admitted; 3 = refused (one line per refusal with a named reason); 2 = setup problem.
# launch_tp6.sh runs this before it starts any engine. The JSON is what `bench/gate/gate.py run --health` records.
# Knobs: PROBE_ARGS="--min-clock-mhz 1800 ..." (thresholds, see tools/health/README.md), PY (python for the probe).
set -eu
. "$(dirname "$0")/common.sh"
OUT=${1:-$G6_HOME/out/rank_health.json}
mkdir -p "$(dirname "$OUT")"
HOSTS_CSV=$(echo "$HOSTS" | tr -s ' ' ',')
# shellcheck disable=SC2086
exec python3 "$REPO/tools/health/rank_probe.py" cluster --hosts "$HOSTS_CSV" --ips "$IPS" --ips2 "$IPS2" \
  --home "$G6_HOME" --proxy-cpu "$PROXY_CPU" --out "$OUT" ${PROBE_ARGS:-}
