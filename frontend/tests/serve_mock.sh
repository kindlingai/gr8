#!/usr/bin/env bash
# Start / stop the mock engine + front end for local tests.
# usage: serve_mock.sh start|stop  (env: PY=venv python, MODEL_DIR=tokenizer dir, PORT=8100, ROUNDS=100,
#   FRONT=front-end binary (default ../bin/g6front-rs), FRONT_ARGS, MOCK_ARGS)
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/tmp/g6venv/bin/python}
MODEL_DIR=${MODEL_DIR:-/tmp/g6tok}
PORT=${PORT:-8100}
SOCK=${SOCK:-/tmp/g6-mock.sock}
FRONT=${FRONT:-$HERE/../bin/g6front-rs}
W=$HERE/tests/work; mkdir -p "$W"
case "${1:-}" in
start)
  "$PY" "$HERE/mock_engine.py" --model-dir "$MODEL_DIR" --sock "$SOCK" --rounds-per-s "${ROUNDS:-100}" \
      ${MOCK_ARGS:-} > "$W/mock.log" 2>&1 & echo $! > "$W/mock.pid"
  for _ in $(seq 50); do [ -S "$SOCK" ] && break; sleep 0.2; done
  [ -x "$FRONT" ] || { echo "no front end at $FRONT: scripts/build_front.sh builds it"; exit 1; }
  (cd "$HERE" && exec "$FRONT" --model-dir "$MODEL_DIR" --engine-sock "$SOCK" --host 127.0.0.1 \
      --port "$PORT" ${FRONT_ARGS:-}) > "$W/front.log" 2>&1 & echo $! > "$W/front.pid"
  for _ in $(seq 50); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 0.2; done
  curl -sf "localhost:$PORT/health" >/dev/null && echo "up on :$PORT" ;;
stop)
  for p in front mock; do [ -f "$W/$p.pid" ] && kill "$(cat "$W/$p.pid")" 2>/dev/null || true; rm -f "$W/$p.pid"; done
  echo stopped ;;
*) echo "usage: $0 start|stop"; exit 2 ;;
esac
