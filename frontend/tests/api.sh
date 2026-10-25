#!/usr/bin/env bash
# api.sh TOKENIZER_DIR [PORT]: tests/test_api.py against the front end and the pseudo engine with the weights in
# tests/api_responses.toml, on this machine (../pseudo-engine/serve.sh says what TOKENIZER_DIR holds). A second front
# end with --allow-thinking-off on PORT+1 runs the test that needs it. The test plays the supervisor on PORT+2.
set -euo pipefail
. "$(dirname "$0")/../pseudo-engine/serve.sh"
pseudo_up "${1:?usage: api.sh TOKENIZER_DIR [PORT]}" "$REPO/frontend/tests/api_responses.toml"
PORT=${2:-8195}; SUP=$((PORT + 2))
front_up "$PORT" --supervisor 127.0.0.1:$SUP
front_up $((PORT + 1)) --allow-thinking-off
cd "$REPO/frontend"
URL=http://127.0.0.1:$PORT SUP_PORT=$SUP python3 -m unittest tests.test_api
URL=http://127.0.0.1:$((PORT + 1)) ALLOW_THINKING_OFF=1 python3 -m unittest tests.test_api.Thinking.test_thinking_off
