# GLP (GGUF) steering input

`--steering-profile` entries load either a schema-v1 profile directory or a **GLP control vector**
(`.gguf`), detected per entry by magic bytes. GLP is the weightless project's format for projective
steering vectors: <https://weightle.ss/>, <https://github.com/msuiche/weightless>, format contract
<https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md>.

A GLP file validates into the same `Profile` as a directory profile, through the same validation
path (`profile_from_parts` in `engine/runtime/steering_profile.h`), so manifest identity, schedule
digest, model matching and unit-norm checks are identical in both containers. Serving, launchers,
request APIs, cache integration and GPU kernels stay separate integrations, exactly as with
directory profiles.

## Conformance gates (fail closed, `engine/runtime/steering_glp.h`)

- `glp.mode` must be `project`. Per the GLP contract an absent key means `add`; both are fatal here.
- `glp.spec_version` 1–3. Above 3 is refused.
- `glp.hook_point` must name the site family the engine applies at. The schema-v1 reference
  operator projects the complete MLP writer output (`scope: full`), so accepted values are
  `residual_stream_post_layer` (which materialises at the combined MLP output on this mhc
  architecture) and `ffn_out_pre_residual`. `attn_out_pre_residual` and the spec-3 hc fold hooks
  (`hc_fold_pre_engram`) name different tensors and are refused — never re-sited silently.
- `glp.rank` must be 1; `glp.dir_scales` (rank-k machinery) is refused.
- Base pin: `glp.base_revision` (or `general.base_model.0.version`) must equal the expected model
  revision. GLP vectors are checkpoint-pinned; a revision mismatch is a hard error.
- `glp.content_sha256` is required and re-verified over the shipped tensor bytes.
- Effective per-layer alpha is `glp.alpha_default * glp.layer_scales[L]` (default 1), and every
  value must fall in the engine's `[0, 4]` band.
- `direction.0` is rejected, widths must be uniform and match the model's hidden size, layer ids
  must be unique, ordered and below the layer count.

## Identity and provenance mapping

The loader synthesizes the schema-v1 manifest deterministically (compact JSON, fixed key order),
so a GLP file has a stable profile identity across boots. Mapping:

- `profile_name`: `general.name` (sanitized), `model`: the expected model identity (the file's own
  name is not trusted for matching), `scope`: `full`.
- `provenance.extraction_tool`: `glp:<glp.generator>`, `provenance.extraction_commit`:
  `glp.content_sha256`, `provenance.fit_sha256`: SHA-256 of the whole `.gguf`.
- GLP files carry no dataset manifests; `dataset_manifests.{train,validation,final}` all carry the
  content hash, which is the digest that commits to the exact direction bytes.

## Example

```sh
# GLM-5.3 GLP-77 (77 layers, alpha 1.0) next to a directory profile:
./gr8 glm53 up --steering-profile /vectors/GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf,/profiles/example
```

The `steering_glp_test` host test covers the gates, synthesis determinism, directory parity and
dispatch; it runs in `scripts/host_tests.sh` and needs no GPU.
