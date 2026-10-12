# Load a local steering profile

A server can load one schema-v1 profile directory containing `manifest.json` and `directions.f32`. Loading is explicit;
without `--steering-profile`, startup and requests retain stock behavior and no steering files are read.

Use the same verified profile on every rank. Local paths may differ. For example, add these options to the normal
`./gr8 glm53 up` command on each box:

```sh
--steering-profile /profiles/example \
--steering-model-revision MODEL_REVISION \
--steering-tokenizer-revision TOKENIZER_REVISION \
--steering-chat-template-sha256 TEMPLATE_SHA256
```

The launcher compares the manifest with the actual resolved weight-set identity and hashes the served
`chat_template.jinja`. `--steering-weight-set` is derived from the model; an explicit different value fails.
`--steering-chat-template` is resolved to the actual served file and passed to the engine, which hashes it again.
If model config metadata is present, dimensions must match GLM-5.3 (6144 hidden, 78 layers) and available revision
metadata must agree. Tokenizer revision metadata is also checked when available. Packs without revision metadata
still require explicit operator revision assertions; the launcher cannot independently recover missing revision history.
The content identities and actual template bytes are checked regardless. Weight hash checks must remain enabled.

Use a self-contained exlr8 release or complete rank-pack set. Mixing external expert weights with separate base packs
is refused because their combined content identity is not represented by this local interface.

The runtime validates the exact manifest/payload hashes, dimensions, normalized directions, schedule and selected
runtime layers before device setup. Missing scheduled layers and duplicate runtime layers fail. Device directions and
shared scratch are initialized once before any layer binding or graph capture; later mutation/reloading is rejected.
Initial warm/capture row scales are positive zero. No per-request profile upload or allocation is added.

The existing all-rank S2 bootstrap compares the loaded profile receipt (manifest digest, name, scope, layers and
forced-mode bits) before readiness; the manifest digest binds the schedule and model identity too. Named requests are
accepted only after the loaded receipt is ready. `ready.s2` remains version 3 without a profile and uses the reserved
version 4 loaded receipt. All ranks must run the same build. The API/shim expose the receipt through `/health`.

## Requests and forced qualification

With a loaded profile named `example`, request `steering_profile: "example"` and an optional `steering_scale` in
`[0,4]` (default 1). Omission or selected scale zero runs stock; unknown names fail even at zero. Exact profile/scale
identity separates stock and selected resident caches. Nonstock cache state is not persisted to stock SSD/remote tiers.
Target prefill, prompt tails, teacher-forced scoring and speculative target verification carry the request scale;
draft rows and padding remain zero. MoE shared scope uses the shared finalizers; full scope uses the general consumes.

`--steering-force SCALE` is an explicit qualification mode for a positive representable FP32 scale up to 4. It requires
a loaded profile, sets the same request identity/scales as normal selection and rejects conflicting explicit requests.
It also selects offline `g6m4 --prompt` plans. This mode is not a replacement for matched serving qualification.
`m4sim` does not expose profile boot flags in this slice and refuses them as unknown arguments.

HF selectors and automatic release discovery are not supported by this local-directory interface. Loading a model
from HF alone does not opt into any profile artifact. Download/materialization support is a separate change.

## Validation limits

The CPU bridge exercises the real C++ preparation path and loaded bootstrap receipt; local launcher tests check
content/template mismatch, malformed/tampered files, force boundaries and path-independent options. A small CUDA
fixture checks exact direction bytes, layer mapping, zeroed initial row scales/shared counters, both scopes, refusal
of missing layers and binding-after-activation errors. Existing projection/finalizer fixtures cover the kernel math.

These are unit checks. They do not establish serving accuracy, stock output parity for an entire boot, speculative
acceptance or throughput. Those results require a matched serving qualification of this activation path.
