#!/usr/bin/env bash
# smoke.sh TOKENIZER_DIR [PORT]: smoketest/run.sh against the front end and the pseudo engine, on this machine
# (serve.sh says what TOKENIZER_DIR holds).
set -euo pipefail
. "$(dirname "$0")/serve.sh"
pseudo_up "${1:?usage: smoke.sh TOKENIZER_DIR [PORT]}" "$PSEUDO/model/responses.toml"
PORT=${2:-8199}
front_up "$PORT"
bash "$REPO/smoketest/run.sh" "http://127.0.0.1:$PORT" glm-5.3
