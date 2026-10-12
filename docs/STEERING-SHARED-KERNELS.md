# Shared-expert finalizers

The general consumes in `STEERING-KERNELS.md` steer the complete writer. Shared scope instead projects only a MoE
layer's shared expert before the rank partial is consumed. Routed expert contributions do not enter its dot product.
This extraction keeps boot activation disabled: real models have empty direction/alpha vectors and null scratch.
Verified profile loading and allocation remain a later change.

## Decode

`tr3_fin_body` handles both an ordinary FP16 shared output and the trellis shared expert. It keeps the existing routed
sum and output conversion. With shared steering selected, each 128-column CTA writes one dot partial; the last of the
48 CTAs sums them in column order and subtracts the correction from the combined local output. The operator is
linear across rank partials. No floating-point atomic reduction or extra launch is introduced.

The selected decode path corrects the already rounded FP16/BF16 partial, then rounds the corrected value again.
The fixture's scalar oracle models that conversion boundary explicitly. This differs from projecting before the first
output conversion by the first conversion's quantization error; it is not an exact FP32 projection of the unrounded
combined row. Prefill projects the shared contribution before its single BF16 output conversion.

Scratch is `[MX][48]` FP32 partials plus `[MX]` unsigned arrival counters. Counters must start at zero. A selected
row resets its counter before returning; disabled rows do not touch scratch. Scratch is reused only across
stream-ordered finalizers, as are the existing plan/control buffers. Overlapping launches need separate scratch.

The arrival protocol follows the [CUDA memory fence rules](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#memory-fence-functions):

- Dot scratch and cross-CTA output reads use volatile accesses to avoid reusing old cache lines.
- Every lane that writes the combined output fences before warp 0 signals the CTA's arrival.
- The correcting CTA reports that all its threads may have written. The existing slot publisher gives each writer
  its own system fence before the final release doorbell. Stock CTAs retain warp 0's existing publication path.

## Prefill

`bindp::k_pf_fin` handles direct FP16 shared outputs and SHX Hadamard/sign/block-scale outputs. It projects the same
rounded FP16 shared values that the stock finalizer adds to the routed sum. Both ordinary BF16 output and direct
reduce-scatter output use this path. Native prefill supplies the existing lane's uniform scale. Full scope continues
to use the general consumes; MoE shared scope selects these finalizers instead.

## Direct unit fixture

On a CUDA 13 GB10 development environment:

```sh
G6_NR=4 EXTRA=-DDEC_S2 TEST=steering_shared_test scripts/c1_tests.sh /tmp/shared-tests
# Or compile just the fixture, without a model:
nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a \
  -DG6_NR=4 -DDEC_S2 -DDEC_SLOT=2 -DPF_RNWIDE=1 -DG6_CP=1 \
  -o /tmp/steering_shared_test engine/runtime/tests/steering_shared_test.cu -lcublas
/tmp/steering_shared_test
```

Compile at TP3/TP6 and without `DEC_SLOT` as well. `PF_FINLOCAL` additionally exercises absent routed prefill positions.
The direct fixture covers both decode shared formats, both output types, routed/absent routes, multi-block experts,
rows across the 8-row tile boundary, all-on/alternating/off/null/zero-alpha guards, repeated arrival-counter reuse,
plan rotation, both slot parities, staging bypass and output/scratch guard rows. A concurrent GPU observer waits on
the actual slot doorbell and snapshots its bytes, which are compared with final output after kernel completion.
Prefill checks direct/SHX, optional block scales, staging/direct RS, uneven ownership and untouched wire padding.

The host uses a scalar Walsh transform and shared-only projection. The output tolerance is
`abs(actual - oracle) < (FP16: 0.0012; BF16: 0.009) * (1 + abs(oracle))`, allowing output quantization and host/device
reduction differences. Disabled rows additionally require exact stock GPU bytes. Nonfinite outputs and CUDA errors fail. These synthetic cases establish no serving
accuracy, timing or throughput result. The observer and synchronization calls are fixture work, not production work.

For exact stock comparison, compile the **same** fixture against the parent and candidate headers; define
`G6_STEERING_TEST_BASELINE` only for the parent. Then run each with `--stock-dump PATH` and compare the files with
`cmp` and SHA-256. Parent and candidate must use identical compiler flags and synthetic data. The normal test run
also checks zero-scale, null direction/scale and zero-alpha guards. CI compiles the fixture even without a GPU;
execution requires a GPU and fails rather than silently skipping it.
