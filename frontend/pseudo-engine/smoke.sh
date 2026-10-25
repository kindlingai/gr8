#!/usr/bin/env bash
# smoke.sh TOKENIZER_DIR [PORT]: smoketest/run.sh against the front end and the pseudo engine, on this machine.
# TOKENIZER_DIR holds GLM-5.3's tokenizer.json, tokenizer_config.json, chat_template.jinja and generation_config.json
# (the release's top level, or a box's model cache tokenizer/ directory). Builds both binaries in release mode first.
set -euo pipefail
TOK=$(cd "${1:?usage: smoke.sh TOKENIZER_DIR [PORT]}" && pwd)
PORT=${2:-8199}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
W=$(mktemp -d)
SOCK=/tmp/g6pseudo.$$.sock   # a socket path must be short (104 bytes on macOS)
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$W" "$SOCK"' EXIT
for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json config.json; do
  [ -f "$TOK/$f" ] && ln -s "$TOK/$f" "$W/$f"
done
ln -s "$HERE/model/responses.toml" "$W/responses.toml"
(cd "$HERE" && cargo build --release --locked --quiet)
(cd "$REPO/frontend/g6front-rs" && cargo build --release --locked --quiet)
"$HERE/target/release/g6pseudo" --model "$W" --sock "$SOCK" &
for _ in $(seq 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
"$REPO/frontend/g6front-rs/target/release/g6front-rs" --model-dir "$W" --engine-sock "$SOCK" --host 127.0.0.1 \
  --port "$PORT" > "$W/front.log" 2>&1 &
for _ in $(seq 100); do curl -sf "localhost:$PORT/health" | grep -q '"ready":true' && break; sleep 0.1; done
bash "$REPO/smoketest/run.sh" "http://127.0.0.1:$PORT" glm-5.3
