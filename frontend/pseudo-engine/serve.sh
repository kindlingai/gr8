# Sourced by smoke.sh and ../tests/api.sh: g6pseudo and the front end on this machine, stopped when the script exits.
#   pseudo_up TOKENIZER_DIR WEIGHTS   build both binaries in release mode, then start g6pseudo with WEIGHTS
#   front_up PORT [ARGS...]           start a front end on 127.0.0.1:PORT against it, and wait until it is ready
# TOKENIZER_DIR holds GLM-5.3's tokenizer.json, tokenizer_config.json, chat_template.jinja and generation_config.json
# (the release's top level, or a box's model cache tokenizer/ directory).
PSEUDO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$PSEUDO/../.." && pwd)
W=$(mktemp -d)
SOCK=/tmp/g6pseudo.$$.sock   # a socket path must be short (104 bytes on macOS)
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$W" "$SOCK"' EXIT

pseudo_up() {
  local tok f
  tok=$(cd "$1" && pwd)
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json config.json; do
    [ -f "$tok/$f" ] && ln -s "$tok/$f" "$W/$f"
  done
  ln -s "$(cd "$(dirname "$2")" && pwd)/$(basename "$2")" "$W/responses.toml"
  (cd "$PSEUDO" && cargo build --release --locked --quiet)
  (cd "$REPO/frontend/g6front-rs" && cargo build --release --locked --quiet)
  "$PSEUDO/target/release/g6pseudo" --model "$W" --sock "$SOCK" > "$W/pseudo.log" 2>&1 &
  for _ in $(seq 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
}

front_up() {
  local port=$1; shift
  "$REPO/frontend/g6front-rs/target/release/g6front-rs" --model-dir "$W" --engine-sock "$SOCK" --host 127.0.0.1 \
    --port "$port" "$@" > "$W/front.$port.log" 2>&1 &
  for _ in $(seq 100); do curl -sf "localhost:$port/health" | grep -q '"ready":true' && break; sleep 0.1; done
}
