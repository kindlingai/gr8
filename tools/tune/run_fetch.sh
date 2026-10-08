#!/bin/bash
# run_fetch.sh RANKS WIDTHS LOG : rerun fetch_ranks.py every 10 min until no layer is skipped (upload still landing).
# Env: HF_TOKEN, GR8 (this checkout), OUT (v2 dir), NR (default 6), LAYERS (default all), PY (default python3).
RANKS=$1; WIDTHS=$2; LOG=$3; D=$(cd "$(dirname "$0")" && pwd)
while true; do
  ${PY:-python3} "$D/fetch_ranks.py" --gr8 "${GR8:?}" --out "${OUT:?}" --nr "${NR:-6}" \
    --ranks "$RANKS" --widths "$WIDTHS" --layers "${LAYERS:-all}" --workers 16 --verify >> "$LOG" 2>&1
  rc=$?
  echo "PASS rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  grep -q "^DONE .* 0 skipped" <(tail -3 "$LOG" | grep DONE) && [ $rc = 0 ] && { echo ALLDONE >> "$LOG"; exit 0; }
  sleep 600
done
