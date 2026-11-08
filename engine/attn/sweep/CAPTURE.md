# Attention index captures (v1)

## What is and is not measured

All original suites (`all`, the legacy spelling `real`, `repro`, `smoke`) are
**synthetic operands on source-derived shapes**, not captured decision rows.
`real` remains an alias for compatibility, not an input-provenance claim.
Uniform top-k now samples **without replacement** and sorts the selected set.
`--suite locality` explicitly adds `--topk-pattern uniform|shared|local` after the
six original shape arguments. These cases have zero prompt weights. The adapter
must support those arguments before they can run.

- `uniform`: independent random unique selections per row.
- `shared`: identical randomly chosen set visible to every row (100% overlap).
- `local`: contiguous most recent visible keys (neighbor overlap follows positions).
- `within_row_unique_fraction`: sum of unique active keys / sum of active keys.
- `adjacent_overlap_fraction`: average intersection / min(active set sizes) for
  adjacent, nonempty rows **in the same cache slot**. Empty/different-slot pairs
  are excluded and `adjacent_pairs` names the sample count. Not all-pairs overlap.

Capture v1 preserves the actual full index matrix in its original row order,
plus exact effective position/session maps. It captures **no Q, window KV,
compressed KV, or sink operands**. Each is explicitly `unavailable` in metadata.
A replay using these indices and generated operands is
**captured-indices + synthetic-operands**, not captured-operands or a calibrated
model decision row. Ring geometry/session layout and phase are observed at the
actual dispatch; knob manifest/revision come from the operator's recorded build
and are explicitly marked `operator_supplied_not_runtime_verified`.

Capture copies/synchronizes/writes on the request path. **Capture runs and their
benchmark JSONs are NOT timing data.** This path does not establish cache-state
fidelity, TTFT savings, numeric fidelity to model operands, or a full prompt sum.
MAIN is **not captured**: only the single-lane prefill callback and its bounded
late replay have a scoped context. MAIN/graph replay/dual-lane callbacks stay
unscoped and do not capture. Full prompt totals remain mixed-source/unvalidated.

## Loader API for the parent adapter

```cpp
#include "capture_io.h"
auto r = sweep_capture::load(path); // throws on malformed metadata/data/hash
// r.metadata, r.topk [n*512], r.positions [n], r.sessions [n]
// r.is_captured: origin == engine; indices only
// r.source_hash: SHA256 of the complete .asw file
// r.metrics: within-row uniqueness + adjacent same-slot overlap
```

Metadata fields: `n`, `prompt_tokens`, `chunk_position` (original prefill chunk,
not the bounded tail's first row), `layer`, `index_source`, `ratio`, `ring_rows`,
`key_rows`, `slot_count`, `request_slot`, `rank`, `heads`, `wl`, `floor`, `ldtk`,
`ck_stride_bytes`, `request_id`, `dispatch_phase`, `format`, `session_form`,
`engine_revision`, `origin`, `knob_manifest`. The serialized record also includes
manifest SHA256, provenance, schema/kind, `timing_data=false`, operand availability.
`request_slot` is the engine request slot; effective implicit addressing uses
`sessions=0` when the engine has already shifted the base to that slot. Never
shift that base twice. Per-row slots and positions must be replayed together.

Adapter argument proposal: six original args followed by `--capture FILE.asw`.
Validate shape/format/phase against the metadata, allocate the recorded ring and
cache strides, upload the complete row-major lists, and report the source hash.
Fail unsupported head counts/session forms/formats rather than coercing them.
Capture values must not be perturbed during transition tests or rotated timing;
use separate synthetic transition checks and distinct captured records/slots.
`capture_tool case` emits descriptors for 16-head packed-CQ/window records,
including per-row slots and noncontiguous positions. Its six shape args use the
original `chunk_position` and exact `key_rows` (zero for window-only); the adapter
must restore effective positions/slots from the record, not those shape args.
It does not claim that a pre-integration adapter supports this. Zero prompt
weights are intentional. Captures have `decision_rows=false` until model-level
calibration/operand and cache-regime gaps are closed.

## Wire format and validation

Single file `.asw`, no sidecars required. Header:

| Offset | Bytes | Meaning |
|---|---:|---|
| 0 | 8 | ASCII `ASWEEP1\n` |
| 8 | 4 | metadata JSON byte length, unsigned little-endian |
| 12 | 4 | payload byte length, unsigned little-endian |
| 16 | 64 | lowercase hexadecimal SHA256(metadata bytes + payload bytes) |
| 80 | metadata length | UTF-8 JSON metadata |
| following | payload length | topk[n][512], positions[n], sessions[n], signed LE32 |

Limits: metadata <= 1 MiB, rows <= 8192, payload exactly `n*514*4` bytes.
Reject truncated/extra data, bad hash/schema/types/geometry, bad padding, duplicate
or unsorted selected indices, negative values except `-1`, out-of-bounds session
or visible key, and wrong active count. Active count is
`min(512, min(key_rows,(position+1)/ratio))`; ratio 0 has all `-1`.
No sorting, deduplicating, row subsampling, or regeneration happens on load.
The file SHA256 identifies the exact metadata and correlated lists; checksums are
integrity checks, not independent proof of who supplied an artifact.

## Build and activation (no launch is performed by these tools)

Normal source has `capture_build.h` set to **0**. Every engine call site and
implementation is preprocessor-guarded. An ordinary build has no capture symbols,
GPU sync, allocation, environment read, or filesystem access from this feature.
Even a capture-enabled build stays off unless `ATTN_SWEEP_CAPTURE_CONFIG` names
a config file. Invalid explicit config/data fails loudly; it never drops back to
invented lists. CUDA graph capture is refused for a scoped capture dispatch.

For a dedicated capture build, either pass `-DGR8_ATTN_SWEEP_CAPTURE=1` to all
relevant TUs, or change the default in `capture_build.h` to 1 in a **separate
recorded capture-only commit**. A fixed remote build wrapper may not forward
local `EXTRA`; do not assume that setting it locally enabled the remote artifact.
After build verify `nm -C <attention object>` contains
`ds41_sweep_capture_indices`; the normal object must not.

```sh
# Host validation; CPU only, no GPU context or service interaction:
python3 -m unittest discover -s engine/attn/sweep -p 'test_*.py' -v
g++ -std=c++17 -O2 engine/attn/sweep/capture_tool.cpp -o /tmp/capture_tool
/tmp/capture_tool inspect /absolute/path/record.asw
/tmp/capture_tool case /absolute/path/record.asw > /tmp/capture-case.json
# Off-ring CUDA object compilation + default-off/on symbol gate (no linking/boot):
python3 engine/attn/sweep/capture_compile_check.py --out out/capture-compile
```

The repository full CPU gate is `./gr8 test`. A failure there is separate from
passing these capture-specific checks; do not summarize a failing full gate as
"all tests passed."

## Bounded initial capture plan

```sh
python3 engine/attn/sweep/capture_plan.py \
  --output-dir /var/lib/m30/attn-capture \
  --engine-revision CAPTURE_BUILD_SHA \
  --knob-manifest ACTUAL_ENGINE_KNOBS.json > capture-plan.json
# On the booked capture artifact only, propagated to the engine process:
# ATTN_SWEEP_CAPTURE_CONFIG=/var/lib/m30/attn-capture/plan.json
```

The output directory must already exist and be writable by the engine UID.
Records use exclusive creation; reruns must use a new directory, not overwrite
old evidence. The same config may be staged on every rank. Default selection:

- rank **0 only**; layers **2, 8, 14, 20, 21**, not all 40 layers;
- prompt sizes 32768 and 131072;
- original chunk positions 0, 16384, 24576, 65536, 122880;
- first matching request per prompt size, later same-size requests ignored;
- <=40 records and <=768 MiB **per process**, checked before D2H;
- index data only, no prompt text/tokens or model operands.

A harness warmup of the same prompt size can be that first request; correlate
saved `request_id` with the engine log rather than assuming the benchmark label.
With W8192, layers 2/8/14/20 over the full chunk and layer21 over tail128, assuming
all selected chunks occur, expected index+position+session payload is:

| Request | Rank0 records | Rank0 payload bytes | Rank1/2/3 |
|---|---:|---:|---:|
| 32K | 15 | 202,894,304 (193.495 MiB) | 0 |
| 128K | 25 | 338,162,656 (322.497 MiB) | 0 |
| Both | 40 | 541,056,960 (515.992 MiB) | 0 |

Add 80 bytes plus serialized metadata per record. Those are calculated storage
estimates, not collected data. A smaller late-final scope, prefix reuse, missing
layers, different chunk size or different capacity can change coverage; inspect
what was actually captured. Selecting all four ranks multiplies totals by four;
the initial rank0 subset is not evidence about inter-rank differences.

Planning estimate: reserve **15 minutes after ready** for the two request runs
and capture I/O, not including build/stage/boot/restore; actual ring overhead is
unmeasured. Owner should put one timeout around the combined request sequence.
If it expires or the expected records are missing, retain the partial artifacts
and report incomplete coverage, not timing or a performance verdict.
