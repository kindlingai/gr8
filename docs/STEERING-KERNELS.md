# General steering projection kernels (inactive serving capability)

This slice of #45 adds the general writer projection to the existing consume kernels:

```text
y = y - (scale * alpha) * direction * dot(direction, y)
residual = residual + y
out = RMSNorm(residual, weight)
```

`scale` comes from each target decode row or the current prefill lane. The immutable direction and alpha
are per layer. A null direction, zero scale, or zero alpha takes the stock arithmetic path. Both attention
and dense MLP writers use the general operator. Full scope also uses it for the complete MoE MLP output;
shared scope leaves that full MLP consume alone pending the separate shared-expert finalizer slice.

The model's direction vectors remain empty on real boots. No profile-loading or force option is added,
so named requests continue to be rejected as documented in [request transport](STEERING-TRANSPORT.md).

## Implementation and numerical correction

`SteerArgs` carries a device direction, optional per-row scales, a uniform scale and alpha. The projection
is part of the existing decode AR consume, prefill AR/RS/wide-RS/owner consume and TP4 QP pair consume.
It precedes the incoming residual addition. Directions for inactive layers are absent.

The source implementation inferred the projected norm by subtracting large squared norms. A regression
with a nearly collinear writer produced `3.16211` instead of the reference normalized tail `1.51911`.
This extraction instead projects the writer, adds the residual, and reduces the squared norm of the
actual resulting row. The regression then produces `1.51953`, within FP16 output tolerance. No
host-precomputed direction norm is needed. The extra reduction/barriers for selected rows are inside
the same consume launch. Selected-path performance has not been qualified for serving.

There is no new per-step CUDA launch, copy, allocation, stream synchronization, or graph node. The
projector uses CTA and cluster barriers within an existing launch. The wide-RS implementation uses its
existing shared reduction storage. Stock rows retain the original add/RMS arithmetic, but register,
shared-memory and branch effects still require performance measurement before an activation release.

## Direct fixtures

```bash
G6_NR=4 EXTRA=-DDEC_S2 TEST=steering_cuda_test scripts/c1_tests.sh <test-bin-dir>
G6_NR=4 EXTRA=-DDEC_S2 TEST=steering_consume_test scripts/c1_tests.sh <test-bin-dir>
```

`steering_cuda_test` retains Jake's helper fixture and adds the near-cancellation regression. It covers
all-off, all-on and alternating rows, exact zero-row identity, projected residual and FP16 normalization.
CUDA API failures and nonfinite outputs fail the test.

`steering_consume_test` includes the actual production `prefill.cuh` and `glue.cuh`. It uses synthetic
fabric slots on one GPU, without model files, a network fabric or a serving boot. It calls the AR,
RS-cluster, RS-wide, owner and both QP pair orders directly, including ranks with no owner rows.
Cases use 1/3/8/32 rows, runtime rank counts 3/4/6, both edge ranks, loopback and real-layout slices,
per-row and uniform scales, zero/null/zero-alpha paths, both scopes and fragment outputs under DEC_SLOT.
It checks untouched owner/guard rows, AG bytes and publication/ticket reset state.

The oracle uses the host `steering::project` contract and computes the norm over the projected residual.
Absolute tolerances are `4e-5` for FP32 residuals and `3e-3` for FP16 normalized values (the latter includes
FP16 rounding). Zero-scale residuals are also compared by bytes. These are finite synthetic cases, not
a bound on all possible activation magnitudes.

For a before/after stock comparison, compile the same fixture source against the parent #74 tree with
`-DG6_STEERING_TEST_BASELINE`, and normally against this tree, using the same CUDA flags. Run each with
`--stock-dump <output-file>` and compare the files with `cmp`. Dumps include residual/output/guard bytes,
AG payloads, doorbell state and AR FP16/BF16 fragment tables. The parent must be pinned in the report.
The baseline build is a test-only mode; it is not a production fallback.

`scripts/build.sh` compiles both fixtures in CI without a GPU and runs them when a GPU is visible. The
normal engine CI still builds every TP width. Cluster output/performance gates are separate; the user's
current instruction limits this extraction to direct unit testing.

## Boundary for later slices

Shared-expert `tr3_fin_body`/`k_pf_fin`, their scratch storage and publication tests remain a separate PR.
Verified model/profile loading and all-rank activation remain later work. This slice intentionally does
not introduce those fields or switches. When transplanting the rest of #45, do not restore the removed
algebraic norm shortcut or the obsolete `SteerArgs.norm_sq`/`Model.steering_norm_sq` fields.
