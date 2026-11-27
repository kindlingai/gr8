#!/usr/bin/env bash
# End-to-end HTTP smoke test for a live g6tts + g6front-rs Drama 3 TTS server (/v1/audio/speech & /v1/audio/voices).
set -euo pipefail
URL="${1:-http://127.0.0.1:8100}"
MODEL_ARG="${2:-}"
echo "# tts smoketest against $URL"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# 1. /health and /v1/models
curl -fsS "$URL/health" | jq -e '.ready == true' >/dev/null
MODELS=$(curl -fsS "$URL/v1/models")
echo "$MODELS" | jq -e '.data | length >= 1' >/dev/null
MODEL_ID="${MODEL_ARG:-$(echo "$MODELS" | jq -r '.data[0].id')}"
if [ -z "$MODEL_ARG" ]; then
  echo "$MODELS" | jq -e '.data[] | select(.id == "drama-3")' >/dev/null && MODEL_ID="drama-3"
fi
echo "1. /health + /v1/models OK: id=$MODEL_ID"

# 2. Buffered WAV synthesis: verify RIFF/WAVE, 44100 Hz, mono, 16-bit, 0.5-10 s, RMS > 0.01
WAV_OUT="$TMP/out.wav"
curl -fsS "$URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"input\":\"Hello there! This is a quick smoke test of the Drama 3 speech engine.\",\"voice\":\"default\",\"response_format\":\"wav\"}" \
  -o "$WAV_OUT"

python3 - "$WAV_OUT" <<'PY'
import math, struct, sys, wave

path = sys.argv[1]
with open(path, "rb") as f:
    hdr = f.read(12)
assert hdr[:4] == b"RIFF" and hdr[8:12] == b"WAVE", f"bad RIFF/WAVE header: {hdr!r}"
with wave.open(path, "rb") as w:
    ch = w.getnchannels()
    sr = w.getframerate()
    sw = w.getsampwidth()
    nf = w.getnframes()
    raw = w.readframes(nf)
assert ch == 1, f"expected mono (1 ch), got {ch}"
assert sr == 44100, f"expected 44100 Hz, got {sr}"
assert sw == 2, f"expected 16-bit (2 bytes), got {sw}"
dur = nf / float(sr)
assert 0.5 <= dur <= 10.0, f"expected duration 0.5-10 s, got {dur:.3f} s"
samples = struct.unpack(f"<{nf}h", raw)
rms = math.sqrt(sum((s / 32767.0) ** 2 for s in samples) / max(1, nf))
assert rms > 0.01, f"expected RMS > 0.01, got {rms:.5f}"
print(f"2. WAV synthesis OK: {sr} Hz mono 16-bit, dur={dur:.2f}s, rms={rms:.4f}")
PY

# 3. Streaming PCM synthesis: verify first body chunk arrives in < 2 s
python3 - "$URL" "$MODEL_ID" <<'PY'
import json, sys, time, urllib.request

url, model_id = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model_id,
    "input": "Streaming PCM latency check for Drama 3.",
    "voice": "default",
    "response_format": "pcm",
    "stream": True,
}).encode("utf-8")
req = urllib.request.Request(
    url.rstrip("/") + "/v1/audio/speech",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
t0 = time.monotonic()
ttfa = None
total = 0
with urllib.request.urlopen(req, timeout=60) as resp:
    assert resp.status == 200, f"unexpected HTTP {resp.status}"
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        if ttfa is None:
            ttfa = time.monotonic() - t0
        total += len(chunk)
assert ttfa is not None and ttfa < 2.0, f"expected first PCM chunk < 2.0s, got {ttfa}"
assert total >= 4096 and (total % 2) == 0, f"unexpected PCM byte count {total}"
print(f"3. Streaming PCM OK: ttfa={ttfa*1000:.1f} ms, bytes={total} ({total/88200.0:.2f}s)")
PY

# 4. Upload voice from returned WAV, list it, synthesize with it, and delete it
VOICE_NAME="smoketest-voice-$$"
python3 - "$URL" "$MODEL_ID" "$WAV_OUT" "$VOICE_NAME" <<'PY'
import base64, json, sys, urllib.request

url, model_id, wav_path, vname = sys.argv[1:5]
base = url.rstrip("/")
wav_b64 = base64.b64encode(open(wav_path, "rb").read()).decode("ascii")

def post_json(path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read()

try:
    st, _ = post_json("/v1/audio/voices", {
        "name": vname,
        "audio": wav_b64,
        "text": "Hello there! This is a quick smoke test of the Drama 3 speech engine.",
    })
    assert st in (200, 201), f"voice upload failed: HTTP {st}"

    with urllib.request.urlopen(base + "/v1/audio/voices", timeout=30) as r:
        listed = json.loads(r.read().decode("utf-8"))
    names = [v["name"] if isinstance(v, dict) else v for v in (listed.get("voices", listed.get("data", [])) if isinstance(listed, dict) else listed)]
    assert vname in names, f"uploaded voice {vname} not found in {names}"

    st, pcm = post_json("/v1/audio/speech", {
        "model": model_id,
        "input": "Cloning voice check.",
        "voice": vname,
        "response_format": "pcm",
    })
    assert st == 200 and len(pcm) >= 4096, f"cloned synthesis failed: HTTP {st}, bytes={len(pcm)}"
    print(f"4. Voice upload + list + cloned synthesis OK: name={vname}, bytes={len(pcm)}")
finally:
    req = urllib.request.Request(base + f"/v1/audio/voices/{vname}", method="DELETE")
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.status in (200, 204), f"voice delete failed: HTTP {r.status}"
print(f"5. Voice delete OK: name={vname}")
PY

# 6. Over-length request returns HTTP 400
OVERLEN_CODE=$(python3 - "$URL" "$MODEL_ID" <<'PY'
import json, sys, urllib.error, urllib.request

url, model_id = sys.argv[1], sys.argv[2]
payload = json.dumps({
    "model": model_id,
    "input": "Hello world.",
    "voice": "default",
    "response_format": "pcm",
    "max_new_tokens": 100000,
}).encode("utf-8")
req = urllib.request.Request(
    url.rstrip("/") + "/v1/audio/speech",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
PY
)
[ "$OVERLEN_CODE" = "400" ] || { echo "FAIL: expected HTTP 400 for over-length request, got $OVERLEN_CODE"; exit 1; }
echo "6. Over-length request HTTP 400 OK"

echo "TTS_SMOKETEST: ALL PASS"
