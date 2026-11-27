#!/usr/bin/env bash
# End-to-end HTTP smoke test for a live Hy4-flash gr8 S2 + g6front-rs server.
set -euo pipefail
URL="${1:-http://127.0.0.1:8080}"
echo "# hy4 smoketest against $URL"

# 1. /v1/models check
MODELS=$(curl -fsS "$URL/v1/models")
echo "$MODELS" | jq -e '.data | length >= 1' >/dev/null
MODEL_ID=$(echo "$MODELS" | jq -r '.data[0].id')
echo "1. /v1/models OK: id=$MODEL_ID"

# 2. Greedy math sanity check
RESP=$(curl -fsS "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Answer with just the number.\"}]}")
TXT=$(echo "$RESP" | jq -r '.choices[0].message.content // .choices[0].message.reasoning_content // ""')
echo "2. Greedy math reply: $TXT"
echo "$TXT" | grep -q "391" || { echo "FAIL: expected 391 in reply"; exit 1; }

# 3. Deterministic repeat check (temperature=0)
RESP2=$(curl -fsS "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"temperature\":0,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"List the first five prime numbers separated by commas.\"}]}")
RESP3=$(curl -fsS "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"temperature\":0,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"List the first five prime numbers separated by commas.\"}]}")
T2=$(echo "$RESP2" | jq -r '.choices[0].message.content')
T3=$(echo "$RESP3" | jq -r '.choices[0].message.content')
[ "$T2" = "$T3" ] || { echo "FAIL: non-deterministic greedy reply"; exit 1; }
echo "3. Deterministic greedy OK: $T2"

# 4. Concurrent 4-session batch sanity check
for i in 1 2 3 4; do
  curl -fsS "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"temperature\":0,\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\"content\":\"Write a short greeting #$i.\"}]}" \
    > "/tmp/hy4-smoke-$i.json" &
done
wait
for i in 1 2 3 4; do
  jq -e '.choices[0].message.content | length > 0' "/tmp/hy4-smoke-$i.json" >/dev/null
done
rm -f /tmp/hy4-smoke-*.json
echo "4. Concurrent 4-session batch OK"
echo "HY4_SMOKETEST: ALL PASS"
