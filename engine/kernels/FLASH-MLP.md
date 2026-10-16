# Flash MLP block (W3)

Compile with `G6_MODEL_FLASH=1`. Geometry comes from `engine/model_geom.h`.
`activation.cuh` clamps gate at `geom::SWIGLU_LIMIT` and up at ±limit **before**
SiLU. Routed, shared, dense-trellis, row-tile, and prefill activation sites use it.
The default GLM expression remains unchanged; the test build's SASS matches the
pre-change build byte for byte.

## Runtime API

`flash_mlp.cuh` includes `bind.cuh` and `glue.cuh` and has no attention dependency.

1. `flash_mlp::load(layer, find, Weights&)` binds W1 names. `find(string)` returns
   `p`, `nd`, `shape[4]`, `bf16`, `fp32`, and `fmt`, like W6's `g6rt::TensorRef`.
   It checks BF16 matrix types/shapes and scans the expert table outside capture.
   It requires `router.bias` to be F32 through the explicit `fp32` flag; a
   non-BF16 tensor is not automatically treated as FP32.
2. Allocate `workspace_bytes(rows)` and use `workspace_at(ptr, rows)`.
   Retain a normal zero-initialized `bind::Scratch` and a `glue::RoutePlan`.
   `bind::init()` sets the kernel shared-memory attributes once.
3. `make(weights, M, fp32_input, bf16_partial, plan, workspace, scratch, block,
   fuse=0)` binds the row count. All pointers refer to device memory. `M` must
   fit the allocated workspace as well as the bind scratch.
4. `launch(block, stream)` computes only the MLP rank partial. W6 owns publication
   to collective `2*i+1`, rank-order all-reduce, and mHC. Pass `sc.part[1]` as the
   output and publish it after launch; do not bind a transient host slot pointer.

The FP32 input is `m.xn2` per design decision D1. The raw BF16 router weights
are converted to FP32 in registers, so the router does **not** first round its
input to BF16. Expert execution casts input to FP16 and uses the existing K1
kernels. Its fragment table lives in the block's extra workspace; it does not
write W6's `xf_x2/xb_x2` tables. Shared experts use W1's trellis row `geom::NEXP`.
Dense layers use lossless BF16 `L###.dense.{gate,up,down}`, FP32 sums/activation,
and one BF16 rounding per rank partial. These scalar-warp dense kernels are a
correctness-first baseline, not a throughput claim.

## Standalone GPU test

Build on the assigned Spark (local x86 compilation only proves compilation):

```sh
nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a \
  -DG6_MODEL_FLASH=1 -DG6_NR=4 -DDEC_S2 -DDEC_SLOT=2 \
  engine/kernels/tests/flash_mlp_test.cu -o /tmp/flash_mlp_test
/tmp/flash_mlp_test --binder-reject-test  # CPU-only type rejection
/tmp/flash_mlp_test --router-check
```

The router check covers M=1,3,8,9,16,32; experts 256–287; exact ties; all-sigmoid
underflow; normalized unbiased weights; every route's slot/column/mask; unused
rows and columns. It requires a GPU.

Prepare real weights on CPU, one rank at a time, without changing the release:

```sh
python engine/ref/flash/check_mlp.py prepare \
  --release RELEASE --widths RELEASE/manifests/exlr8-manifest-k4.json \
  --dump engine/loader-rs/target/release/g6plandump \
  --golden GOLDEN/prompt-bf16.safetensors --layer 0 --rows 32 --out CASE
# MoE: --golden GOLDEN/layer3-K4.safetensors --layer 3. Repeat K2/K3 with matching manifests.
python engine/ref/flash/check_mlp.py run --case CASE --binary /tmp/flash_mlp_test
python engine/ref/flash/check_mlp.py compare --case CASE
```

Move CASE to the Spark before `run`; return its outputs before `compare`, or
pass the correct local golden path with `--golden`. The compact safetensors keep
W1 payload bytes and dtypes exactly; `.meta` contains only validated tensor offsets.
The test loads one rank at a time. It emits M=1,3,8,9,16,32 outputs up to `--rows`.
Use `--offset 32` for the second half of a 64-token golden.

`run/compare --fuse 1` or `2` also checks the fused K1 path. For a forced fused
launch at every M, run with `G6_K1_FUSE_POLICY=0`; the default runtime policy can
otherwise choose separate fc1/fc2. The test compares actual-router block output
and a second expert output from the **identical W0 router logits**. Router logits,
router selection/weights, and fixed-route expert output have separate gates.
Thus a router near-tie cannot hide which op diverged.

CPU-only checks:

```sh
python engine/ref/flash/test_activation.py
python engine/ref/flash/test_mlp_fixture.py
python engine/ref/flash/test_mlp_fixture.py --dense-case CASE
```

CPU checks prove activation math, exact fixture bytes and the real W1 dense TP4
shard sum versus W0. They do not prove GPU execution. See the external W3 evidence
report for the current measured status; no GPU or performance result follows
from a successful compile.
