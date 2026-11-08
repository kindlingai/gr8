# Attention sweep interface v1

Standalone microbenchmark, not a served-engine change. Baseline source: `1ed4501c`.
All committed kernel code is ours. Do not copy external implementations.

## Variant contract (stable)

Create `engine/attn/sweep/variants/<name>/variant.cuh`; name matches
`[a-z][a-z0-9_]*`. Additional private headers may live in that directory.
The bench includes `../ds41_attn.cuh` before your header. Implement:

```cpp
namespace sweep_variant {
// Pointer to the actual CUDA kernel, used for cudaFuncGetAttributes.
inline const void* kernel(const ds41a::AttnP& p);
// Unique substring of that kernel's MANGLED ptxas entry name (not demangled).
inline const char* ptxas_key(const ds41a::AttnP& p);
inline size_t dynamic_smem(const ds41a::AttnP& p);
// Called once per case OUTSIDE timing. Set function attributes here.
inline cudaError_t init(const ds41a::AttnP& p);
// Enqueue one complete attention operation, asynchronous on the given stream.
inline void launch(const ds41a::AttnP& p, cudaStream_t stream);
}
```

`AttnP` is defined at `engine/attn/ds41_attn.cuh:628`. Only `p.out` may
be written. All input buffers, positions, masks, dimensions, sink, scale,
compressed format and top-k order remain unchanged. No global knob changes,
allocations, host synchronization, reference reads, or hidden setup per launch.
Change the implementation and launch geometry inside the variant only; do not
edit shared engine headers or the bench. One kernel per launch in v1, except the approved prewiden pre-pass below
(resource columns refer to the main kernel). For unsupported cases, dispatch the original
kernel and report its pointer/key/smem. In particular the built-in pipe/swz
variants fall back to off for window-only and floor>0 cases, as serving does.

Start from `variants/off/variant.cuh`, or the pipe/swz launch adapters. A custom
kernel can include a privately copied-and-modified version of OUR kernel in
its directory, with a distinct symbol/namespace. It must not redefine ds41a.

## EXACT and reference

Off = `k_attn_cpa_t<false,true>` for CQ; `<false,false>` for window;
`<true,true>` for CQ with bounded-replay floor. No split kernel. Every binary
contains the independent original off reference plus exactly one variant.
The reference and candidate receive identical immutable inputs. Full output
`n * 16 * 512 * sizeof(bf16)` is compared by `memcmp`; differing **bytes** are
counted. EXACT requires zero differing bytes, not a float tolerance, including
signed zero. Output buffers are initialized to distinct poison patterns before
comparison. A mismatching variant still produces a row; it is not a winner.

## Cases and weights (source-derived, synthetic values)

TP4, one prompt, 16 local heads, head dim 512. All four ranks have identical
shapes (different head values); do NOT multiply latency by four.
Sources at baseline commit:
- `engine/model_geom_ds41.h:20,31-44,54-59`: 40 main layers, 64 total
  heads, 512 dims, window 128; layers 0..1 window, 2..19 ratio 2,
  20..39 ratio 1. `engine/attn/ds41_attn_plan.h:85-102`: local head binding.
- `engine/runtime/ds41_v2.h:14-15,30-32`: W=8192, late boundary=21,
  tail=128. `engine/runtime/ds41_v2_attn.cuh:188`: core uses ALL chunk
  rows, not only a rank's owned rows.
- `engine/runtime/s2/ds41_prefill.h:5-7,71-80`: first N-1 tokens are
  headless prefill; last token is a MAIN row. Thus final chunk is **8191**,
  not 8192, for exact 32768/131072-token prompts.
- `engine/runtime/ds41_pf_split.h:5-12,55-68`: layers 21..39 run only
  the last 128 rows with window floor at that tail's first position.
  `:179-187` documents optional late-final skipping; report configuration.
- `engine/attn/ds41_attn_ops.cuh:445-485` and
  `engine/attn/ds41_attn.cuh:1862-1876`: device positions imply decode-form
  ring even for the chunk at zero; WL=128, TK=512 (or 0 for window),
  `floor=0` except late-tail cases. Top-k is ascending with -1 tail.

For N=32768 or 131072, enumerate chunk starts `s=0,8192,...,N-8192`;
`n=min(8192,N-1-s)`. At EACH start:

| case | rows | first position | ratio | floor | calls/chunk |
|---|---:|---:|---:|---:|---:|
| window | n | s | 0 | 0 | 2 |
| CQ r2 | n | s | 2 | 0 | 18 |
| CQ r1 early | n | s | 1 | 0 | 1 |
| CQ r1 late | 128 | s+n-128 | 1 | s+n-128 | 19 |

Then MAIN row at N-1: 1 row, floor 0, window/r2/r1 weights 2/18/20.
Default estimate assumes late-final skipping OFF. `--late-final` gives late
rows zero weight except chunks with fewer than 256 remaining prefill rows,
per `ds41::br::late_skip`. Counts exclude DSpark draft work and everything
other than main-model attention core; these are NOT TTFT estimates.

The two prompt case sets are deduplicated; each case carries separate 32K
and 128K weights. Summation: `sum(mean_us * calls)/1000`, no rank multiplier.
The explicit `repro` suite adds 8192 rows at pos 24576, r2/r1/window with
32768 compressed rows (the old profile harness fixture), zero prompt weight.
The `smoke` suite adds WL=37 and WL=100 mixed window/CQ boundary cases,
zero prompt weight; real full chunks have aligned WL=128 (no mixed tile).
Default `all` runs real shapes plus repro and boundary checks.

Values are deterministic synthetic data reused from our
`engine/attn/tests/ds41_comp_fp4_test.cu` fixture methods, seed 12345 reset
per case. Shapes/masks are engine-derived, not claimed as captured tensors.
Timings are CUDA-event mean and sample SD over reps; each rep times `iters`
consecutive launches and divides by iters. Warmup is outside timing.

## Build/run (script entry point)

From repository root (Python 3 stdlib, CUDA toolkit, C++17):

```sh
# Local RTX 5090, native x86 compiler; only run on an available GPU.
python3 engine/attn/sweep/run.py --arch 120a --nvcc /usr/local/cuda/bin/nvcc \
  --variants off pipe swz --suite smoke --out /tmp/attn-sweep-local
# On a cleared GB10, native ARM compiler (never copy an x86 executable):
python3 engine/attn/sweep/run.py --arch 121a --nvcc /usr/local/cuda-13.0/bin/nvcc \
  --variants off pipe swz --suite all --warmup 3 --reps 5 --iters 20 \
  --out /root/attn-sweep/run-UNIQUE
# Build all independently, parallel; run sequentially, even if a sibling fails.
# Omit --variants to discover every variants/*/variant.cuh.
# --build-only / --run-only separate stages. --suite repro isolates known row.
```

Compiler: `nvcc -std=c++17 -O3 -gencode arch=compute_ARCH,code=sm_ARCH
-DG6_MODEL_DS41=1 -I engine -Xptxas -v -DSWEEP_VARIANT_HEADER=\"...\"
engine/attn/sweep/bench.cu -o OUT/VARIANT`. No fast-math flag.
Build logs and raw case JSONL persist beside binaries; summary CSV/Markdown
contains time, SD, percent vs off, byte comparison, regs, STACK, spill stores,
spill loads, dynamic smem. Missing ptxas data is UNKNOWN, never assumed zero.

The harness prints free/total/required bytes and an explicit LOW_MEMORY
message and exits 77 when headroom is insufficient. Script retries (default
3 attempts, bounded delay) then writes LOW_MEMORY rows, not a silent success.
A broken variant cannot prevent other binaries from building/running.
Do not stop engines to create memory; the run owner schedules the GPU gap.

## Generic driver / per-kernel adapter (Chuck's reusable-bench requirement)

`run.py --kernel adapter.json --cases cases.json --reference off
--variants-dir PATH` is the reusable entry point. Defaults select attention CQ.
The driver knows NO attention dimensions. Adapter JSON supplies `name`, `header`,
`variants_dir`, `case_generator` (optional Python file), and `defines` (list).
Paths are relative to the JSON file. CLI overrides replace name (`--kernel-name`),
case file, reference directory name, or variants directory. A case list is a JSON
array of `{id, args: [strings or numbers], weights: {"32K": count, "128K": count}}`;
args are opaque to the generic driver. Cases retain their input order.
For attention, args are `[rows, pos, ratio, floor, WL, key_rows]`, ratio 0 means
window-only. A user-provided case file needs no generator and no driver edit.

The generic CUDA binary includes the configured adapter header, then the
reference variant header renamed to namespace `sweep_reference`, then the
candidate header as `sweep_variant`. A variant header must use that namespace
token, not stringify it. Shared helper headers should use normal include guards.
The attention variant signature above is also expressible as `sweep_adapter::Params`.
Future kernels replace the adapter, not the timing/compare/report loop:

```cpp
namespace sweep_adapter {
using Params = /* kernel-specific launch parameter type */;
struct Fixture;  // owns host/device inputs and two independent output buffers
size_t required_bytes(const std::vector<std::string>& args);
Fixture make_case(const std::vector<std::string>& args, uint64_t seed);
Params params(Fixture&, bool reference); // selects output only
void* output(Fixture&, bool reference); // contiguous device output
size_t output_bytes(const Fixture&);
void destroy(Fixture&);
}
```

Generic binary handles memory preflight, CUDA events, output poison/memcmp,
resource queries, and JSON output. Adapter constructs/frees inputs and implements
shape validation (throw `std::runtime_error` for invalid args). Each variant's
`kernel`, `ptxas_key`, `dynamic_smem`, `init`, `launch` takes that adapter's Params.
Selecting another reference changes comparison baseline, clearly named in reports;
attention's acceptance bar in this campaign remains `--reference off`.

## Approved prewiden exception (tribune ruling)

`init(p)` may allocate variant-owned scratch once per case outside timing,
using a static/namespace pointer sized from p. Release at process exit or the
next init; no allocation in launch. Besides p.out, only this scratch may be
written. `launch(p, stream)` may enqueue the pre-pass then main kernel on
that stream; CUDA events cover BOTH. `kernel()`/`ptxas_key()` describe the
main kernel. The variant's report must separately list pre-pass regs, STACK,
spills from its ptxas entry and scratch bytes. On allocation failure return
cudaErrorMemoryAllocation from init, so the generic runner retries.

For the reference binary itself the harness aliases sweep_variant to
sweep_reference instead of including the same header twice, so variant headers
may use `#pragma once`. The runner sets `SWEEP_SAME_VARIANT` automatically.

Shared 5090 GPU runs use the common lock (builds may remain parallel):

```sh
python3 engine/attn/sweep/run.py --arch 120a --build-only --out /tmp/my-sweep
flock /tmp/attn-sweep-5090.lock python3 engine/attn/sweep/run.py \
  --arch 120a --run-only --out /tmp/my-sweep
```

Allocation OOM after preflight (another process races the free-memory check)
also exits 77 and is retried, including adapter fixture allocations.

## Revision 2 hardening (supersedes the repeated-address v1 timing description)

**Synthetic screening only.** No kernel table alone is a model keep/drop gate.
Locality-sensitive ideas require real index correlation and a validated cache
regime; 128K totals require same-request per-rank trace calibration. Capture of
indices alone does not make Q/KV/sink captured operands. No blanket numeric
"haircut" is inferred from one old pipe run; surrounding work/critical path and
paired ring row/TTFT gates must be measured separately.

The harness holds at least three independent output slots and `--pools M`
input pools (default3;1 is allowed). `init` runs only at shape initialization,
not per invocation. Before each correctness call and each timed pair-group,
the adapter refreshes synthetic input values and snapshots immutable buffers.
Every candidate invocation gets freshly poisoned output. Sampled TIMED calls
are compared to a freshly computed reference on their current inputs. Negative
controls live outside `variants/`, under `negative_controls/`; compute-once
must produce timed mismatches, and off-stream launch must fail enqueue proof.

Adapter revision2 adds:
```cpp
Params params_with_output(Fixture& inputs, Fixture& output_slot, bool reference);
void refresh_inputs(Fixture&, uint64_t nonce, cudaStream_t);
void snapshot_inputs(Fixture&, cudaStream_t);
size_t check_integrity(Fixture&, cudaStream_t); // nonzero corruption count
```
Captured-index cases preserve captured indices; operand freshness remains
synthetic unless separately captured operands are explicitly provided. Future
full captured replay must rotate recorded input snapshots, not silently perturb
them. Integrity errors are separate from output-byte mismatches.

Each process times BOTH reference and candidate, alternating order. It emits
paired per-repetition deltas and off/off null measurements (nominal ±3% band),
plus raw candidate/reference samples. The driver separately estimates between-
process paired uncertainty; a narrow within-process SD does not remove drift.
Very short kernels can have larger null variation and are not individual speed
claims. `--flush-mb N` adds explicitly labelled synthetic cache-displacement
traffic OUTSIDE kernel timing; it does not claim to replay model predecessors.
`--pools 4` rotates four independent pools; zero flush is not proof of a model
cache state. Allocation/poison/refresh/check overhead is outside timed kernels.

Launch checks capture the supplied nonblocking CUDA stream, inspect its kernel
nodes, and require the reported main `kernel(p)` to appear. `cudaStreamQuery`
is checked for valid success/not-ready states and both are counted: a completed
short kernel can legitimately return success, so not-ready is not alone an
enqueue proof. Candidate-private scratch remains the variant author's duty;
its metadata must list scratch bytes and every pre-pass kernel's resources.

### Explicit per-chunk setup / amortization

Existing launch adapters keep per-call semantics (their pre-pass inside launch
remains charged EVERY call). An explicitly amortized adapter defines
`SWEEP_VARIANT_HAS_SETUP 1` and supplies:
```cpp
inline void setup_chunk(const sweep_adapter::Params& p, cudaStream_t stream);
```
`init` may allocate scratch but must not hide input-derived work. `setup_chunk`
is timed once per group of `--amortize M` calls on unchanged input; its time is
charged as setup/M in each call's mean. Every call still has a freshly poisoned,
rotating output. Input refresh and a new setup occur between groups. A variant
without this opt-in gets NO implicit amortization of its current launch. Report
`has_chunk_setup`, M, and the proposed engine reuse unit; synthetic amortization
is not proof that the model can reuse that state for M calls.

### Attention geometry and captured indices (revision 2)

The six positional shape arguments remain compatible. Optional **pairs** after
those arguments are `--ring-rows R`, `--slots S`, `--slot I`, `--phase PHASE`,
and `--topk-pattern uniform|shared|local`. Defaults are a **synthetic single-slot,
rowcap8192 configuration**: ring8319, slots1, slot0; phase is late_replay for a
nonzero floor, main for n1, otherwise prefill_v2. Key rows remain the explicit
sixth argument, not an inferred real cache capacity. Generated weights describe
that conditional source-derived workload, NOT the running boot manifest.

The capacity no longer shrinks with a tail's n. Actual engine allocation is in
`engine/attn/ds41_attn_ops.cuh:335-346` (rowcap and shared/per-slot rings),
`:361-376` (compressed allocation capacity/2+2 and per-slot stride), and
`:473-481` (shared prefill binding). Multislot boot MAIN uses its actual decode
ring capacity, not the synthetic default; supply flags or a capture. The served
pipe enable bracket is `engine/runtime/g6m4.cu:1789-1802`: built-in pipe/swz now
fall back to the exact off dispatch in MAIN, legacy prefill and late replay.
They only use the optimized kernel for phase=prefill_v2, TK>0, floor=0.

Alternatively supply **only** `--capture /absolute/path/file.asw` after the six
arguments. The adapter validates the capture hash/geometry, requires those six
arguments to match metadata, and preserves original row order, positions,
per-row slot IDs and top-k. It uses the recorded ring/key capacities and phase;
packed CQ and window-only records with16 heads are supported. Other formats
fail closed. Capture flags cannot be mixed with geometry overrides. JSON
`adapter_metadata` records the file SHA256, provenance, phase, capacities and
unique/adjacent-overlap metrics. A synthetic-test record stays labelled as such.

These are **index-only captures**. Q, packed KV and sink remain fresh synthetic
values; window operands are finite FP8-E4M3-valued BF16, still synthetic, not
learned/captured. The unused BF16 key shadow is omitted by default. Numeric tests may opt in
with `--shadow-keys 1`; JSON reports it and memory preflight includes it. Capture/locality rows have zero prompt
weights and `decision_rows=false` until same-request trace and cache-regime
validation. Index correlation alone is not an end-to-end model decision gate.

`test_adapter_gpu.py --bundle BUILD_DIR --out NEW_DIR` checks noncontiguous
captured-test row order and per-row sessions, MAIN/legacy fallback, local-index
stress, guards and refusal of mismatched capture geometry. The test acquires
the campaign GPU lock itself. It does not run a served engine or collect real
model data.
