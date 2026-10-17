#!/usr/bin/env bash
# The gate for --allow-thinking-off (see frontend/README.md "Forced-disabled thinking").
#
# Runs the maintainer's probe against a deployed head: count 1 to 200, one number per line, digits only,
# temperature 0, thinking off (--allow-thinking-off must be set on the front end), N runs (default 8).
# A run passes when the RAW reply is exactly the numbers 1..200, one per line, digits only, in order.
# No digit-stripping, no truncation: a 201st number, prose anywhere, bullets, blank lines between
# numbers or CRLF all fail; only trailing whitespace (the usual final newline) is tolerated. Prints
# PASS/FAIL with per-run metrics and a summary; the flag has a home either way, but a failing probe
# must carry the caveat in the docs. Set COUNT200_RAW_DIR=<dir> to save each run's raw reply.
#
# usage: scripts/count200_probe.sh http://<head>:8100 [runs] [served-name]
set -euo pipefail
BASE=${1:?usage: count200_probe.sh http://<head>:8100 [runs] [served-name]}
RUNS=${2:-8}
NAME=${3:-glm-5.3}

pass=0
for i in $(seq "$RUNS"); do
  body=$(jq -n --arg m "$NAME" '{model:$m, max_tokens:1024, temperature:0, stream:false,
    chat_template_kwargs:{enable_thinking:false},
    messages:[{role:"user", content:"Count from 1 to 200, one number per line, digits only."}]}')
  out=$(curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d "$body")
  # Strict comparison of the raw reply (the maintainer's ask): the reply must be exactly the numbers
  # 1..200, one per line, digits only, in order. No digit-stripping, no truncation: a 201st number,
  # prose anywhere, bullets, blank lines between numbers or CRLF all fail. Only trailing whitespace
  # (the usual final newline) is tolerated before the byte-exact comparison.
  want=$(seq 1 200)
  content=$(printf '%s' "$out" | jq -j '.choices[0].message.content // ""')
  if [ -n "${COUNT200_RAW_DIR:-}" ]; then
    mkdir -p "$COUNT200_RAW_DIR"
    printf '%s' "$out" | jq -j '.choices[0].message.content // ""' > "$COUNT200_RAW_DIR/run$i-raw.txt"
  fi
  nlines=$(printf '%s' "$content" | awk 'END{print NR+0}')
  ndigit=$(printf '%s\n' "$content" | grep -cE '^[0-9]+$' || true)
  stripped=$content
  while [[ $stripped == *[[:space:]] ]]; do stripped=${stripped%?}; done
  if [ "$stripped" = "$want" ]; then
    echo "run $i: PASS  (lines=$nlines, digit-only=$ndigit/200)"
    pass=$((pass + 1))
  else
    echo "run $i: FAIL  (lines=$nlines, digit-only=$ndigit/200)"
    printf '%s' "$content" | sed -n '1,3p' | sed 's/^/    first: /'
    printf '%s' "$content" | tail -n 3 | sed 's/^/    last:  /'
  fi
done
echo "$pass/$RUNS clean"
[ "$pass" = "$RUNS" ]
