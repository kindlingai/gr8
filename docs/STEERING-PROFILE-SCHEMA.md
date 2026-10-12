# Steering profile schema v1

This change provides host-only types, validation and a CPU reference operator. Serving, launchers, request APIs,
cache integration and GPU kernels do not consume them yet. `load_profile` compares the manifest to a caller-provided
`ExpectedModel`; verifying that identity against the actual served files belongs to boot integration.

A profile is a non-executable directory with exactly the two runtime inputs below. Extra files are ignored.

```text
PROFILE/
  manifest.json
  directions.f32
```

`manifest.json` has this shape:

```json
{
  "schema_version": 1,
  "profile_name": "example",
  "model": {
    "name": "glm-5.3",
    "revision": "exact-model-revision",
    "weight_set": "64 lowercase hex digits identifying the exact model set",
    "tokenizer_revision": "exact-tokenizer-revision",
    "chat_template_sha256": "64 lowercase hex digits",
    "hidden_size": 6144,
    "layer_count": 78
  },
  "scope": "full",
  "layers": [2, 3, 4],
  "alphas": [1.0, 1.0, 1.0],
  "directions": {
    "file": "directions.f32",
    "dtype": "float32-le",
    "byte_count": 73728,
    "sha256": "64 lowercase hex digits"
  },
  "provenance": {
    "extraction_tool": "publisher-defined extraction tool",
    "extraction_commit": "git commit",
    "dataset_manifests": {
      "train": "64 lowercase hex digits",
      "validation": "64 lowercase hex digits",
      "final": "64 lowercase hex digits"
    }
  }
}
```

## Validation

The profile name matches `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. `scope` is `shared` or `full`. Model name, revision and
tokenizer revision are nonempty strings. `weight_set` is a 64-character lowercase hexadecimal model-set identity;
`chat_template_sha256` is a digest in the same format. Dimensions are positive uint32 values. Every model field must
match the corresponding `ExpectedModel` field exactly.

Layers form a nonempty, strictly increasing list of zero-based model-layer indices below `layer_count`. `alphas`
has the same length and every value is finite and in `[0, 4]`, checked before conversion to FP32.

`directions.f32` stores one little-endian IEEE-754 FP32 vector of width `model.hidden_size` for each layer, in manifest
order. Every component is finite. Each vector's FP64-computed norm must be in `[0.999, 1.001]`. `byte_count` is exactly
`4 * hidden_size * len(layers)` and `sha256` covers the complete file. Short and trailing bytes are rejected.
The manifest is limited to 1 MiB and the directions payload to 64 MiB; sizes are checked before allocation.

The manifest must be UTF-8 JSON. Duplicate and unknown keys are rejected in every schema object, as are missing
required fields. Numeric grammar is strict: leading zeroes, NaN, infinity and overflow are rejected. Integer fields
require integer tokens (`1`, not `1.0` or `1e0`) in the field's range. The JSON reader's integer accessor accepts only
magnitudes at most 2^53 without rounding. Nesting is bounded; strings validate UTF-8 and escaped surrogate pairs.
Number parsing is independent of the process locale. Finite numeric underflow to zero is accepted.

Provenance requires nonempty `extraction_tool` and `extraction_commit`, and SHA-256 digests for the `train`,
`validation` and `final` dataset manifests. Optional `capture_manifest_sha256` and `fit_sha256` must also be digests.
Optional `family` is null, `global-band` or `per-layer`; optional `direction_layer` is null or an in-range layer index.

## Identity and scale contracts

The profile identity is SHA-256 of the **exact manifest bytes**, including whitespace and newlines. Those bytes
commit to the directions digest, model, scope, schedule, alphas and provenance. Reformatting a manifest changes its
profile identity. The schedule digest hashes a one-byte scope (`shared=1`, `full=2`) followed by, for each selected
layer, its uint32 index and the FP32 alpha bits, both in little-endian order.

A request scale must be finite and in `[0, 4]` **before** conversion to FP32. Negative zero becomes positive zero;
a positive value that underflows FP32 becomes zero. A tiny negative input and an input just above 4 remain invalid
even if FP32 rounding would put them in range. The canonical FP32 bits are the request/cache identity. Scale zero
selects the off identity. A named profile defaults to scale 1; a nonzero scale without a name is rejected. The host
resolver also rejects unknown names, selection without a loaded profile, and explicit requests conflicting with a
forced selection. These are contracts for later API integration, not active request options.

The cache namespace is 70 bytes: enabled (1), manifest digest (32), scope (1), schedule digest (32), scale bits (4,
little-endian). The off namespace is all zeroes. Later cache and replicated-state integration will consume these
helpers; this change does not enable steered cache reuse or rank agreement checks.

## CPU mathematical reference

For a writer output row `y`, unit direction `d` and effective alpha `request_scale * layer_alpha`, `project` computes
`y -= effective_alpha * dot(y, d) * d`, accumulating the dot product and correction in FP64 before FP32 stores.
Effective alpha zero performs no reads or writes through either pointer. `shared` applies to the shared-expert
writer output; `full` applies to the complete MLP writer output. The reference does not project the residual stream.
GPU implementations and their comparison against this reference are separate changes.

## CPU tests

The shared acceptance table is `engine/runtime/tests/fixtures/steering_contract.json`. Later Python and Rust
validators should consume the same cases for profile names, scales and JSON syntax. The C++ tests also cover model
identity, manifest and tensor integrity, malformed inputs, request resolution, namespace separation and projection.
On a Linux host with g++ (no GPU required):

```sh
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I engine/runtime \
  engine/runtime/tests/steering_test.cpp engine/loader/g6pack.cpp -o /tmp/steering_test
/tmp/steering_test engine/runtime/tests/fixtures/steering_contract.json
```

`scripts/host_tests.sh` includes this test before the tests that need Python and Rust. The existing pack loader
supplies SHA-256 and requires Linux when linking this test.
