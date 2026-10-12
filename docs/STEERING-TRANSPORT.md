# Steering request and row transport

This is the inactive transport foundation extracted from #45. Real `g6m4` boots have no loaded steering
profile and publish the existing `G6S2 3` readiness line. No profile-loading or force CLI/config switches
are introduced here. Named HTTP requests fail with a clear 400, including named scale-zero requests.
Directions and projection kernels are not installed by this change.

## Request contract

Both completion endpoints recognize `steering_profile` and `steering_scale`. Missing/null fields are absent.
Names use the [profile schema](STEERING-PROFILE-SCHEMA.md). Scale defaults to 1 with a name, must be a
finite number in [0,4], and is rounded to FP32; negative zero becomes positive zero. Scale-only zero is
stock, while a nonzero scale requires a profile. A name must match a loaded capability even at scale zero.
Forced-mode fixtures accept omitted fields or the same profile/scale and reject explicit conflicts.

Rust sends the exact FP32 value as a JSON number; the shim independently validates it, then appends
`steer <name> <eight lowercase hex FP32 bits>` after the existing `ins` field in the S2 request header.
The C++ parser resolves it against immutable boot identity. The request, grant, session and driver payload
carry manifest digest, scope, schedule digest, and scale bits together. Budget continuations retain the
profile and scale. Cancelled and retired slots clear their payload and row state before reuse.

## Capability and coordinated version change

The reserved loaded receipt is:

```text
G6S2 4 <epoch> <ranks> <pool_rows> <mtp> <guard> <max_rows> <max_lp> <profile> <manifest-digest> <scope> <active-layers> <forced> <decimal-scale-bits>
```

The shim validates its name, digest, scope, layer count and forced scale, then exposes `hello.steering`
as documented in [PROTOCOL.md](../PROTOCOL.md). No-profile readiness stays version 3. `/health` keeps its
readiness status and now returns JSON containing `ready` and `steering`; disabled capability has
`enabled: false, forced: false`. Loaded reports include only the first 12 characters of the manifest digest.

The internal replicated control codec changes from **G6S2E4 to G6S2E5**, including stock grants, because
grants now carry identity. Encoder and decoder change together and reject old, truncated or malformed
frames. Deploy the same engine build on every rank; do not mix E4 and E5. Scheduler agreement hashes
include boot and live-request identities. This is independent of the Unix JSON protocol's version 1.

## Cache and model inputs

At the service boundary, identity maps to #72's generic 70-byte `CacheNamespace`. Cache hooks receive
opaque bytes; the allocator has no steering dependency. Stock is all zero. Count, match, begin and retain
all use the same key, including zero counts for nonstock admissions to clear pending stock metadata.
Nonstock resident reuse stays excluded from stock persistent tiers.

Every target MAIN row carries its request's exact FP32 scale, including prompt logprobs, prompt tails,
inserted tokens, and target verification of MTP/DFlash2 drafts. Draft/chain and padding rows carry positive
zero. Both target prefill lanes refresh their scalar on every quantum, including stock after a selected
request. `StepIn` extends the existing control transfer and `k_step_in` stages the scale array; no extra
per-step CUDA launch or transfer is added. The scale buffer is not yet consumed by projection kernels.

## CPU fixtures

- Rust sampling tests use the shared name/scale table, wire tests check omission and exact scale bits,
  and generation tests cover preservation across a thinking-budget continuation.
- `frontend/tests/test_steering.py` checks shim parity. With `STEERING_TRANSPORT_CHECK` pointing to the
  compiled `steering_transport_test`, real shim bytes pass through the C++ parser and replicated grant
  codec for every valid shared scale case. CI requires that bridge.
- `steering_transport_test.cpp` covers disabled/forced parsing, malformed and truncated control frames,
  mixed rows, draft/padding zeroes, cancellation/slot reuse, namespace state and payload/grant disagreement.
- `steering_state_test.cpp` covers scheduler identity hashes, grant round-trip and exact resident cache keys.
- Existing shim concurrency/insert tests and Rust tool-parser tests remain in their suites.

The mock's `--steering` capability is synthetic and does not change model output. These tests establish
transport behavior; serving numerics and performance require later qualification when kernels are wired.
