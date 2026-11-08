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
edit shared engine headers or the bench. One kernel per launch in v1 (resource
columns refer to that kernel). For unsupported cases, dispatch the original
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
  tail=128. `engine/runtime/ds41_v2_attn.cuh:187`: core uses ALL chunk
  rows, not only a rank's owned rows.
- `engine/runtime/s2/ds41_prefill.h:5-7,71-80`: first N-1 tokens are
  headless prefill; last token is a MAIN row. Thus final chunk is **8191**,
  not 8192, for exact 32768/131072-token prompts.
- `engine/runtime/ds41_pf_split.h:5-12,55-68`: layers 21..39 run only
  the last 128 rows with window floor at that tail's first position.
  `:174-181` documents optional late-final skipping; report configuration.
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
