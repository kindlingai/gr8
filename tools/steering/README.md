# Steering profile artifact tools

`release.py` validates, packs and materializes a finished steering profile using only Python's standard library.
Profile extraction belongs to the publisher. These tools implement the [schema-v1 host contract](../../docs/STEERING-PROFILE-SCHEMA.md);
serving, launcher flags, profile discovery and GPU steering are separate integrations.

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
