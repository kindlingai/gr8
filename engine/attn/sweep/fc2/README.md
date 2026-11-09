# fc2 sweep bench (adapter 2)

Our reusable sweep bench (`engine/attn/sweep/`), second adapter: the served v2 routed-expert **down projection**

    mxtc::k_fc2_tc<576, 5120, 4, 2, 8, 8, 4>      engine/kernels/ds_moe_tc.cuh:455-607
    launched once per column group (4 launches)   engine/runtime/ds41_v2_ffn.cuh  down_group(), args()

The bench runs the served kernel (`off`) and one variant on identical **synthetic** inputs and compares the result
byte for byte. All code here is ours. Nothing is taken from pico, spark-ds41f, vLLM or TileLang.

## What is timed

Only the four `k_fc2_tc` launches (one per column group, `a.col_group = 0..3`). **Not timed:** the plan build
(`k_import_routes`, `k_scan_ro`, `k_place_routes`), the shared-expert GEMM, `k_combine_group`. So a number here is
smaller than the "all groups incl. combine" number from the ncu campaign. It is also not the 9.90 ms stock
full-width call (that one is `COL_GROUPS=1`, `os2=16`, fused combine). See the report for the cross-check.

## Fixture

`M` tokens (1..8192), top-6 routes per token without replacement over `--experts N` (6..384, default 384).
The plan is built by the engine's own kernels in the served order. Weights and activations are synthetic bytes.

    ./bin ... -- 8192 --pattern uniform|skew [--holes 0|1] [--experts 384]

* `uniform`: routes spread evenly. `skew`: power-law expert popularity (long expert tails).
* `--holes 1`: some routes are cut (`idx = -1`), so tiles are partly live (mask rows inside tiles).

Suites (`cases.py`): `repro` = `fc2_m8192_uniform` (the served call); `all` = repro + skew, M=8191, holes, M=2048,
M=128; `smoke` = three tiny shapes that fit a busy 5090.

## Exactness

`k_fc2_tc` does **not** write rows whose `tile_mask` bit is clear. So the compare is by row:

* live rows (mask bit set in the INPUT plan): byte for byte equal to the reference (`memcmp`, so signed zero counts);
* dead rows: must still hold the harness poison. A variant that writes a dead row is a MISMATCH.

The column `compare_mode` says which rule was used. Attention (full-output `memcmp`) is unchanged.

## Add a variant (copy me)

    cd engine/attn/sweep/fc2
    cp -r variants/tmpl variants/NAME
    sed -i 's/tmpl/NAME/g' variants/NAME/*          # renames the symbol, namespace and ptxas key
    $EDITOR variants/NAME/kernel.cuh                 # change the kernel here; keep the launch contract

`NAME` must match `[a-z][a-z0-9_]*`. Rules for `variants/NAME/variant.cuh`:

1. `kernel(p)`, `ptxas_key(p)`, `dynamic_smem(p)`, `init(p)`, `launch(p, stream)` as in `variants/off`.
   `ptxas_key` is a unique substring of the **mangled** ptxas entry name. It must match exactly one
   `Function properties for` line, or the regs/spill columns stay empty.
2. `launch` enqueues exactly the four column-group launches. Write only `p.a.ypart`. Everything else in `p.a`
   is the immutable fixture. No allocations, no host sync, no global knob changes.
3. Keep `#define SWEEP_VARIANT_HAS_BLOCK_THREADS 1` and `block_threads()` (the block size you launch with) to get
   the occupancy columns. Change `dynamic_smem` and `block_threads` when you change the tiling.
4. If you change template parameters, change them in `kernel()`, `ptxas_key()` and `launch()` together.
   The grid must still cover `n_slots * gy2 * os2` blocks per group unless your kernel's tiling says otherwise.

`test_fc2_adapter.py` keeps `variants/tmpl/kernel.cuh` identical to the engine kernel (line for line, except the
symbol name). When the engine kernel changes, re-copy it.

## Run

Local 5090 (small shapes only; ~3.9 GB free when the serving stack is up):

    python3 engine/attn/sweep/run.py --kernel engine/attn/sweep/fc2/fc2.json --arch 120a --suite smoke \
        --variants off NAME --out /tmp/fc2-run --build-only
    flock /tmp/attn-sweep-5090.lock python3 engine/attn/sweep/run.py --run-only --out /tmp/fc2-run \
        --warmup 1 --reps 3 --iters 3 --pools 3

GB10 (sm_121a, full M=8192): `fc2/gb10_run.sh [VARIANT ...]`. It builds, takes `/tmp/attn-sweep-gb10.lock`, and
measures with `--timeout 900`. It was **not run** by its author.

Output: `summary.csv` / `summary.md`, `results.jsonl`, `run-manifest.json`.

## Columns added for this adapter (also valid for attention)

| column | meaning |
|---|---|
| `regs`, `local_bytes` | from `cudaFuncGetAttributes` on the variant kernel |
| `ptxas_regs`, `stack_bytes`, `spill_store_bytes`, `spill_load_bytes` | from the ptxas `-v` log, selected by `ptxas_key` |
| `dynamic_smem`, `static_smem` | bytes per block (dynamic from the variant, static from the kernel attributes) |
| `block_threads` | threads per block (variant opt-in) |
| `max_active_blocks_per_sm` | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for that kernel, block size and dynamic smem |
| `occupancy_pct` | `max_active_blocks_per_sm * block_threads / max_threads_per_sm` (1536 on sm_120/121) |
| `compare_mode` | `live-rows-exact-dead-rows-pristine` or `full-output-memcmp` |

## Tests

    python3 -m unittest discover -s engine/attn/sweep/fc2 -p 'test_fc2_adapter.py'   # CPU, sm_120a + sm_121a compile, GPU
    FC2_SKIP_GPU=1 python3 -m unittest ...                                           # skip GPU (also skipped if <3 GB free)

Negative controls live in `negative_controls/` and **must never be scored as candidates**. Each one must be caught:

| control | what it does | expected |
|---|---|---|
| `ulp_flip` | flips 1 bit of one live fp16 output | MISMATCH, `mismatch_bytes == 1` |
| `dead_row_write` | writes one row that must stay unwritten | MISMATCH |
| `compute_once` | computes in the first launch only | `timed_mismatch_bytes > 0` |
| `skip_group` | launches 3 of 4 column groups | MISMATCH |

## Limits

* Inputs are synthetic. Timings are kernel-only. They do not include combine, the shared GEMM or the plan build.
* Routing is uniform or power-law, not captured from a prompt. `input_provenance` is `synthetic` in every row.
* One variant per binary. Variants run in balanced orders over several process rounds (see `run.py --rounds`).
* The null band is 3 %. A gain below that is not a result.
