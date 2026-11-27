#!/usr/bin/env bash
# End-to-end HTTP smoke test for a live g6embed + g6front-rs /v1/embeddings server.
set -euo pipefail
URL="${1:-http://127.0.0.1:8100}"
MODEL_ARG="${2:-}"
echo "# embed smoketest against $URL"

# 1. /health and /v1/models
curl -fsS "$URL/health" | jq -e '.ready == true' >/dev/null
MODELS=$(curl -fsS "$URL/v1/models")
echo "$MODELS" | jq -e '.data | length >= 1' >/dev/null
MODEL_ID="${MODEL_ARG:-$(echo "$MODELS" | jq -r '.data[0].id')}"
echo "1. /health + /v1/models OK: id=$MODEL_ID"

# 2. Single text embedding (unit-normalized finite vector)
R1=$(curl -fsS "$URL/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"input\":\"A dog chasing a ball in the park.\"}")
DIM=$(echo "$R1" | jq '.data[0].embedding | length')
[ "$DIM" -ge 256 ] || { echo "FAIL: unexpected embedding dim $DIM"; exit 1; }
echo "$R1" | python3 -c '
import json, math, sys
d = json.load(sys.stdin)
v = d["data"][0]["embedding"]
assert all(math.isfinite(x) for x in v), "non-finite embedding"
norm = math.sqrt(sum(x * x for x in v))
assert abs(norm - 1.0) < 1e-3, f"expected unit norm, got {norm}"
assert d["usage"]["prompt_tokens"] > 0
'
echo "2. Single text embedding OK: dim=$DIM, unit-normalized"

# 3. Batched array input ordering + semantic cosine sanity
R2=$(curl -fsS "$URL/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"input\":[\"A dog running in a sunny park.\",\"A puppy playing outdoors on grass.\",\"TCP congestion control window size and packet loss.\"]}")
echo "$R2" | python3 -c '
import json, sys
d = json.load(sys.stdin)
assert len(d["data"]) == 3
assert [x["index"] for x in d["data"]] == [0, 1, 2]
e0, e1, e2 = (x["embedding"] for x in d["data"])
cos = lambda a, b: sum(x * y for x, y in zip(a, b))
sim_near, sim_far = cos(e0, e1), cos(e0, e2)
assert sim_near > sim_far + 0.10, f"expected near ({sim_near:.3f}) > far ({sim_far:.3f}) + 0.10"
print(f"3. Batched array + semantic sanity OK: near={sim_near:.4f} > far={sim_far:.4f}")
'

# 4. Matryoshka dimensions truncation + base64 encoding
R3=$(curl -fsS "$URL/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"input\":\"Matryoshka dimension test\",\"dimensions\":512,\"encoding_format\":\"base64\"}")
echo "$R3" | python3 -c '
import base64, json, math, struct, sys
d = json.load(sys.stdin)
raw = base64.b64decode(d["data"][0]["embedding"])
assert len(raw) == 512 * 4, f"expected 2048 bytes, got {len(raw)}"
vals = struct.unpack("<512f", raw)
norm = math.sqrt(sum(x * x for x in vals))
assert abs(norm - 1.0) < 1e-3, f"expected unit norm at d=512, got {norm}"
print("4. MRL dimensions=512 + base64 OK")
'

# 5. Multimodal image + text messages payload (synthetic 64x64 solid PNG via stdlib zlib)
IMG_URI=$(python3 -c '
import base64, struct, zlib
w, h = 64, 64
raw = b"".join(b"\x00" + b"\x40\x80\xc0" * w for _ in range(h))
def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
print("data:image/png;base64," + base64.b64encode(png).decode())
')
R4=$(curl -fsS "$URL/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"$IMG_URI\"}},{\"type\":\"text\",\"text\":\"Describe this blue square.\"}]}]}")
echo "$R4" | python3 -c '
import json, math, sys
d = json.load(sys.stdin)
v = d["data"][0]["embedding"]
assert all(math.isfinite(x) for x in v)
norm = math.sqrt(sum(x * x for x in v))
assert abs(norm - 1.0) < 1e-3, f"expected unit norm, got {norm}"
print(f"5. Multimodal image+text OK: dim={len(v)}, prompt_tokens={d[\"usage\"][\"prompt_tokens\"]}")
'

echo "EMBED_SMOKETEST: ALL PASS"
