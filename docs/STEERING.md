# Runtime steering

GR8 can load one verified immutable rank-one profile at boot and select it independently for each request. Loading is
explicit. Starting with no profile keeps steering disabled; an HF release's bundled profiles are ignored unless the
operator selects one. Loading a profile leaves requests stock until they select it or explicit forced mode is enabled.

## Load a profile

Keep the normal [Quick start](../README.md#quick-start) model, topology and server options. With a self-contained HF
exlr8 release, add these options on every rank:

```sh
./gr8 glm53 up --model https://huggingface.co/ORG/REPO@COMMIT --model-cache /model-cache \
  --steering-profile hf:default
```

`hf:default` takes the release index's default; `hf:NAME` selects a named artifact. The same verified identity must be
selected on every rank. Each box may use a different local cache path. Follow [HF loading](STEERING-HF.md) for the
index format, download checks, offline restart, identity limitations and publication steps. Gated repositories use the
existing `HF_TOKEN` authentication. A model URL alone never selects a profile.

For a local schema-v1 profile directory, add:

```sh
--steering-profile /profiles/example \
--steering-model-revision MODEL_REVISION \
--steering-tokenizer-revision TOKENIZER_REVISION \
--steering-chat-template-sha256 TEMPLATE_SHA256
```

The launcher derives the actual weight-set identity and served template file. Supplied identities are constraints;
wrong values fail. Available dimensions/revision metadata must agree, while unavailable revision history remains an
operator/publisher assertion. The engine independently hashes the served template and validates the profile before
device setup. Weight hash checks must stay enabled. Use a self-contained release or complete rank-pack set; mixed
external experts/base packs are refused. See [local loading](STEERING-LOCAL.md) for direct-engine requirements.

The [schema](STEERING-PROFILE-SCHEMA.md) defines the exact manifest/directions bytes, dimensions, layer schedule,
unit directions and alphas. The engine materializes directions once before layer binding/graph capture, initializes
scales/shared scratch, and compares the loaded receipt across ranks before readiness. Invalid artifacts, absent
scheduled layers, identity/template mismatches and inconsistent rank receipts fail before requests are served.
`GET /health` reports the loaded capability, profile identity/scope/layer count and forced mode.

## Select per request

Both completion endpoints accept these optional fields after a profile named `example` has been loaded:

```json
{"steering_profile":"example","steering_scale":1.0}
```

A named profile without a scale uses 1. Omit both fields for stock behavior. A scale without a name is accepted only
at zero and canonicalizes to stock. A named zero scale also runs stock, but the name must still be the loaded profile.
Unknown profiles and nonzero selection on a server without a profile return a request error. Scale must be finite and
in `[0,4]`; canonical FP32 bits identify the request and its cache namespace.

`--steering-force SCALE` explicitly selects the loaded profile for every request and offline `g6m4 --prompt` plan.
It accepts positive representable FP32 scales up to 4 and rejects conflicting explicit requests. It is a qualification
mode; without it, loading alone never selects requests. `m4sim` does not expose profile boot flags.

Stock and selected resident prefix caches are separated by exact manifest/scope/schedule/scale identity. Selected
state is not persisted into stock SSD/remote tiers. Target decode, prefill, prompt tails, teacher-forced scoring and
speculative target verification carry the selection; draft models and padding remain unselected. All ranks must run
the same engine build and agree on the loaded receipt.

## Operator and precision

For selected writer output `y`, unit direction `r`, layer alpha and request scale, the intended operation is:

```text
y' = y - request_scale * layer_alpha * r * (r^T y)
```

It applies to attention writer output and the configured MLP writer output. `full` scope uses the complete MLP writer;
`shared` MoE scope takes its dot product from the shared contribution, excluding routed experts. A profile derived for
post-layer residual projection is a different operator and must be converted before use.

Projection is fused into existing consumes/finalizers; directions are uploaded at boot and row scales travel in the
existing control transfer. Inactive rows take the stock branch before direction loads/projection arithmetic. Actual
results include each path's FP16/BF16 conversion boundaries: the shared decode finalizer corrects a previously rounded
combined partial, while prefill projects before its final output conversion. See the
[general kernel contract](STEERING-KERNELS.md) and [shared finalizer precision/publication contract](STEERING-SHARED-KERNELS.md).
Those differences prevent treating every path as an exact FP32 projection of an unrounded row.

## Qualification and provenance

[Artifact tools](../tools/steering/README.md) pack a publisher-provided finished profile deterministically and preserve
its exact manifest bytes. Extraction stays with its publisher. The [offline report tools](../tools/steering/REPORTS.md)
summarize frozen paired results and normalized CUDA operation counts. They do not collect measurements, validate
external judges or establish end-to-end behavior/accuracy/performance.

The extracted contracts, request transport, kernels, local boot and HF loading have unit/compile/CI checks. Cluster
and serving gates were waived for this split. That waiver supplies no new matched whole-boot numerical, throughput,
speculative acceptance or model-behavior measurement. A deployment should keep its measured qualification tied to the
exact engine, model, profile, dataset and topology; historical contributor results cover their recorded source heads.

GR8 bundles no profile or model-specific behavior claim. Profiles should record their source artifacts, provenance,
qualification and applicable license/notices. The engine and tools remain under this repository's MIT license;
model weights and profiles carry their own terms. Contributor credit is retained in Git and the original PR history.
