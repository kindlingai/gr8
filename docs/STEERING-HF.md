# Load a steering profile from a model release

An HF model release may publish schema-v1 steering artifacts. Loading the model alone keeps steering disabled and
does not fetch its steering index or artifacts. To select one explicitly, add this to the normal launch on every box:

```sh
./gr8 glm53 up --model https://huggingface.co/ORG/REPO@COMMIT --model-cache /model-cache \
  --steering-profile hf:default
```

`hf:default` selects the release index's `default_profile`; `hf:NAME` selects its named record. The model must be a
self-contained HF exlr8 release, without `--base-packs`. A local directory still uses the
[local-profile interface](STEERING-LOCAL.md). `--steering-force` has the same explicit qualification semantics for
both sources. Keep weight hash checks enabled.

The launcher derives revision assertions from the artifact and binds them to the actual resolved weight-set identity
and served chat template through the same validation used for a local profile. Available config dimensions and
revision metadata must agree. Explicit `--steering-model-revision`, `--steering-tokenizer-revision`,
`--steering-chat-template-sha256` or `--steering-weight-set` values act as additional constraints; different values fail.
Where model packs/configs lack revision history, the artifact's revision strings remain assertions by its publisher;
they are not independently recovered from the weights. The quant release's HF commit is distinct from those base-model
revision strings. The engine independently validates the materialized profile and hashes the served template before
device setup, then compares loaded receipts across ranks before readiness.

## Release layout

Publish a deterministic artifact with the existing standard-library helper:

```sh
python3 tools/steering/release.py pack /verified-profile /release/steering/example.safetensors
```

Add `/release/steering/index.json` with exactly these fields:

```json
{
  "schema_version": 1,
  "default_profile": "example",
  "profiles": [
    {
      "name": "example",
      "file": "steering/example.safetensors",
      "sha256": "ARTIFACT_SHA256",
      "byte_count": 123456
    }
  ]
}
```

Replace the digest and byte count with the actual artifact's values. Each name starts with an ASCII alphanumeric
character and contains at most 128 alphanumeric, dot, underscore or hyphen characters. Names are unique, the default
must be listed, and each path must be exactly `steering/NAME.safetensors`. `hf:default` is the default-selection alias.
The embedded profile name must match the index. No arbitrary paths or cross-repository artifact URLs are accepted.
The index is limited to 1 MiB and an artifact to the existing 1 MiB header plus 64 MiB payload limits and its 8-byte
header length. Bounds and published file sizes are checked before downloads. The index, artifact digest/byte count,
embedded manifest, directions and model binding must all validate before rank preparation.

The runtime image already includes `tools/steering/release.py` and the launcher; no additional Python package or
model-specific profile is bundled. The artifact carries model-specific provenance and qualification separately.

## Cache and offline restart

Profiles live under the resolved HF commit's cache directory:

```text
<model-cache>/hf/<org>--<repo>/<commit>/steering/
  index.json
  .index-identity.json
  example.safetensors
  profiles/example/manifest.json
  profiles/example/directions.f32
```

The first selection verifies the index against the pinned release and records its digest, size and commit. A cached
restart rechecks that record, the index, the selected artifact and the materialized files without calling the hub.
The index identity record protects against accidental cache corruption; the model cache is not an authenticity trust
store against a user who can rewrite it. Cache the release/ref metadata and model files too for a full offline boot.
`prepare` refreshes a moving model revision; a new commit gets its own profile cache and reselects its default.

Only the selected artifact is downloaded. A missing materialized directory is rebuilt from a verified cached artifact,
preserving the exact UTF-8 manifest and direction bytes. A conflicting/corrupt existing directory fails; it is not
silently replaced. A commit-scoped file lock serializes concurrent profile downloads/publication. Missing cold-cache
files still require network access. A corrupt index, artifact or materialized directory must be removed from the local
cache before it can be fetched or reconstructed again.

## Validation limits

Fake-hub tests cover opt-in behavior, named/default selection, strict bounds and index fields, hash/name/payload and
model/template mismatches, concurrent publication, offline rematerialization and refresh on the same options object.
The real C++ boot bridge accepts the downloaded materialization and refuses a changed served template. These unit
checks and runtime-image import checks establish packaging and validation, not model-specific behavior, serving
accuracy, throughput or speculative acceptance. Matched serving qualification remains separate.
