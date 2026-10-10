#!/bin/bash
# run_fetch.sh RANKS WIDTHS LOG : rerun fetch_ranks.py every 10 min until it succeeds (an upload still landing fails it).
# Env: HF_TOKEN, OUT (v2 dir), NR (default 6), LAYERS (default all), PY (default python3).
RANKS=$1; WIDTHS=$2; LOG=$3; D=$(cd "$(dirname "$0")" && pwd)
while true; do
  ${PY:-python3} "$D/fetch_ranks.py" --out "${OUT:?}" --nr "${NR:-6}" \
    --ranks "$RANKS" --widths "$WIDTHS" --layers "${LAYERS:-all}" --workers 16 --verify >> "$LOG" 2>&1
  rc=$?
  echo "PASS rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  [ $rc = 0 ] && { echo ALLDONE >> "$LOG"; exit 0; }
  sleep 600
done
