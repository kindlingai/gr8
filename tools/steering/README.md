# Steering profile artifact tools

`release.py` validates, packs and materializes a finished steering profile using only Python's standard library.
`contrast.py` makes a profile from pairs of contrasting answers ([below](#make-a-profile-from-contrastive-answers)).
These tools implement the [schema-v1 host contract](../../docs/STEERING-PROFILE-SCHEMA.md); serving, launcher flags,
profile discovery and GPU steering are separate integrations.

## Commands

From the repository root:

```sh
python3 tools/steering/release.py pack profiles/example artifacts/example.safetensors
python3 tools/steering/release.py unpack artifacts/example.safetensors materialized/example
```

The source directory contains `manifest.json` and `directions.f32`. Extra source files are ignored. `pack` validates
both files before atomically replacing the output artifact. Input paths can be `str` or `os.PathLike` objects in the
Python API. The CLI reports input and validation errors on stderr and exits with status 2.

`unpack` validates the complete artifact before publishing a staged directory in a single rename. The destination
must be absent, empty, or already contain the exact same manifest and payload. Repeated materialization of identical
bytes reuses the directory; conflicting contents and symlink destinations are rejected. Use a distinct directory for
each profile identity. No extraction path comes from artifact metadata.

## Artifact and identity

The safetensors file contains exactly one `F32` tensor, `directions`, with shape `[len(layers), hidden_size]` and
contiguous data offsets `[0, payload_bytes]`. The little-endian tensor bytes are copied directly. Its metadata is:

- `format`: `gr8-steering-profile-v1`
- `gr8_manifest`: the **exact UTF-8 input manifest text**, including whitespace and final newlines
- `manifest_sha256`: SHA-256 of those exact UTF-8 bytes

The enclosing header uses sorted, compact ASCII JSON padded with spaces to a multiple of eight bytes. Packing the
same two input files produces identical artifact bytes. Unpacking preserves both input files byte for byte, so it
also preserves the C++ loader's manifest identity; repacking that directory reproduces the same artifact. Different
manifest formatting intentionally produces a different identity and artifact. The tool never canonicalizes the
embedded manifest or adds a newline.

The reader rejects extra tensors/fields, malformed JSON or UTF-8, non-finite numbers, invalid shapes/offsets/types,
truncation, trailing payload bytes, and manifest/payload digest mismatches. It applies the schema's ordered layer,
model identity, provenance, alpha and finite unit-direction checks. Manifest and safetensors header limits are each
1 MiB; the payload limit is 64 MiB. Header escaping adds overhead, so a manifest within its own limit can still be too
large to pack. Readers bound allocations and require integer JSON tokens for integer fields.

Digests detect corruption and commit to bytes; they do not authenticate a publisher. Model identity is compared to a
complete caller-provided expected identity when supplied. Binding that identity to the actual served model files
remains boot integration's responsibility.

## Python API

```python
from tools.steering import release

manifest = release.validate_directory('profiles/example', expected=expected_model)
release.pack('profiles/example', 'artifacts/example.safetensors')
manifest, directions = release.read('artifacts/example.safetensors')
release.unpack('artifacts/example.safetensors', 'materialized/example')
```

`expected_model` contains all seven fields of the schema's `model` object: `name`, `revision`, `weight_set`,
`tokenizer_revision`, `chat_template_sha256`, `hidden_size` and `layer_count`. Partial identities are rejected.
Validation failures raise `ProfileError`; file operations can also raise `OSError`. No third-party Python packages,
network access, model checkpoint or GPU are required.

## Tests

```sh
python3 -m unittest discover -s tools/steering/tests -v
```

The standard-library suite covers exact-byte round trips, deterministic repacking, shared JSON/name/alpha fixtures,
invalid artifacts, failed publication, concurrent packing and the CLI. To include Python/C++ acceptance and digest
parity on Linux, compile the test bridge and supply its path:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I engine/runtime \
  engine/runtime/tests/steering_profile_check.cpp engine/loader/g6pack.cpp -o /tmp/steering_profile_check
STEERING_PROFILE_CHECK=/tmp/steering_profile_check python3 -m unittest discover -s tools/steering/tests -v
```

The CPU CI entry point builds the bridge and runs the complete suite. Running Python alone skips only the C++ parity
test. The shared fixtures live at `engine/runtime/tests/fixtures/steering_contract.json`; scale cases exercise the
manifest alpha bounds here, with request-scale canonicalization covered by the host contract tests.

## Make a profile from contrastive answers

A pairs file holds one JSON object per line: a prompt and two answers to it. `positive` shows the behaviour to steer
toward and `negative` the behaviour to steer away from. Keep everything else about the two answers the same (topic,
length, format), since the profile learns whatever separates them. An optional `system` adds a system turn.

```json
{"prompt": "Why is the sky blue?", "positive": "天空之所以是蓝色的……", "negative": "The sky is blue because ..."}
```

`contrast.py` teacher-forces both answers through the engine and records each layer's writer outputs over the answer
tokens. The direction of a layer is the unit difference of the positive and negative mean outputs. The alpha moves the
negative mean's component along it onto the positive one's, `alpha = 1 - c+ / c-`. The operator shrinks
(`0 < alpha < 1`) or flips (`alpha > 1`) a component but cannot grow one, and the schema caps alpha at 4, so a layer
qualifies only where alpha lands in `[0, 4]`. `fit` prints every layer's numbers and which it kept. Profiles are
`full` scope.

1. Tokenize against the running server. This renders each (prompt, answer) with the served chat template and finds
   where the answers start and end:

   ```sh
   python3 tools/steering/contrast.py tokenize pairs.jsonl work --api http://HEAD:8100 --model glm-5.3
   ```

2. Stop serving. Copy `work/` to the same path on every rank. On each rank, turn that rank's last supervisor
   `engine:` line into an offline command (no `--serve`, MTP, prefill lanes, drafter, KV tiers or steering; one
   generated token per sequence):

   ```sh
   python3 tools/steering/contrast.py argv out/supervisor.log work > work/offline.sh
   ```

3. Run the offline command on every rank, workers first, from the tree root. Rank 0 also gets `G6_RESID_DUMP`, about
   3.9 MB per token for GLM-5.3, so stream it into the reducer through a fifo (`reduce` needs numpy):

   ```sh
   mkfifo work/dump
   python3 tools/steering/contrast.py reduce work < work/dump &
   G6_RESID_DUMP=work/dump sh work/offline.sh
   ```

   `reduce` checks that the dump's row count matches `work/seqs.json` and writes `work/sums.npz`.

4. Fit, taking the model identity from any profile manifest made for the same served model:

   ```sh
   python3 tools/steering/contrast.py fit work profiles/my-profile --base OTHER/manifest.json --name my-profile-v1
   ```

   `--layers 10-60` limits the candidates. `--strength 0.5` goes half way from `c-` to `c+`. A request's
   `steering_scale` multiplies the alphas again at serving time.

Each layer's alpha closes the whole gap on its own, and the layers add up, so a full-strength profile overshoots.
Sweep the request scale and read the answers. Scales past the good range make the output mix both behaviours, then
break the text.

The operator scales a component and never adds one, so a profile pushes toward the positive side only by shrinking
or flipping what is there. On a layer where the two means sit on opposite sides, alpha is above 1 and the operator
flips the component, whichever side the input starts on. Swapping positive and negative gives nearly the same
operator, not the opposite one.

### Anchored profiles

`fit --anchor` makes a one-way profile. Each layer's direction is the part of `negative - positive` orthogonal to the
positive mean, and alpha is `--strength` (default 1). The profile then removes what sets the negative answers apart
and leaves positive-like input almost untouched. `--toward negative` swaps the roles, so the same `sums.npz` gives a
profile for each direction:

```sh
python3 tools/steering/contrast.py fit work profiles/en-zh --base OTHER/manifest.json --name en-zh-a1 --anchor
python3 tools/steering/contrast.py fit work profiles/zh-en --base OTHER/manifest.json --name zh-en-a1 --anchor \
  --toward negative
```

Erasing a behaviour does not pick a replacement. With only two behaviours in the pairs, some answers can land on a
third one, such as another language.

### Example: English and Chinese

[examples/en-zh](examples/en-zh) makes English-to-Chinese and Chinese-to-English profiles. `make_pairs.py` asks the
served model for an English and a Chinese answer to each question. `score.py` measures the share of Chinese in
held-out answers at several scales. `heldout-zh.txt` holds the same held-out questions in Chinese.

Measured on GLM-5.3 with 48 pairs and `reasoning_effort` medium, 16 held-out questions per language. A cell is the
Chinese share of the answers, with the number of mostly-Chinese answers (share above 0.8) in brackets:

| profile | layers | question | scale 0 | 0.5 | 0.65 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|---|
| en-zh, default fit | 52 | English | 0.00 (0) | 0.67 | 0.66 (9) | 0.25 | |
| | | Chinese | 0.99 (16) | 0.99 (16) | 0.99 (16) | 0.48 (2) | |
| en-zh-a1, `--anchor` | 78 | English | 0.00 (0) | 0.95 (15) | | 0.99 (16) | 0.99 (16) |
| | | Chinese | 0.99 (16) | 1.00 (16) | | 0.99 (16) | 1.00 (16) |
| zh-en-a1, `--anchor --toward negative` | 78 | English | 0.00 (0) | 0.00 (0) | | 0.00 (0) | 0.00 (0) |
| | | Chinese | 0.99 (16) | 13 Chinese, 3 English | | 13 English, 2 Spanish, 1 Russian | 14 English, 2 Spanish |

The default fit breaks down past 0.65 (0.11 at scale 2). The anchored profiles hold up to 1.5, and English questions
under zh-en-a1 at 1.5 give nearly the same text as no steering. With the template's default effort the long English
reasoning pulls answers back toward English, and repeated runs vary.

## Offline qualification reports

`evaluate.py` summarizes frozen paired evaluation rows; `trace_audit.py` compares normalized CUDA operation counts.
Both use only the standard library. See [REPORTS.md](REPORTS.md) for their input contracts, reproducibility settings,
exit status and limits. These reports complement the [runtime integration guide](../../docs/STEERING.md); they do not
run a model or replace matched serving qualification.
