# Adaptive reduced attention sweep (owner-run GB10 recipe)

Source base:02a6e14e plus the adaptive/reduced/sharding commits. The runner defaults
remain fixed-round; opt in with `--adaptive-ci 1.0`. See ORDER.md for the nominal
CI/null gate and its sequential-look/position-balance limitations.

## Build ONLY on an approved GB10 builder, then distribute

This follows our `work/scripts/gn1win.sh` container pattern. It is a recipe, not
a deployment performed by the bench author. The build machine must be GB10 /
AArch64; local x86-host cross-arch CUDA objects are not runnable AArch64 binaries.
The **host does not need gcc/g++**: pico-build:20261106 supplies the compiler.

The owner archives the desired committed checkout into a NEW source directory
under `/root/attn-sweep/` on the booked builder. Substitute that absolute path
for SOURCE and a NEW output directory for BUILD below. Both must be visible
through the mount. Do not edit sources while building.

```sh
SOURCE=/root/attn-sweep/src-adaptive-COMMIT
BUILD=/root/attn-sweep/adaptive-reduced-build-COMMIT

docker run --rm --network none \
  -v /root/attn-sweep:/root/attn-sweep -w "$SOURCE" \
  pico-build:20261106 \
  python3 engine/attn/sweep/run.py \
    --nvcc /usr/local/cuda-13.0/bin/nvcc --arch 121a --jobs 4 \
    --variants off pipe swz splitbar ctune_pipe_cap ctune_pipe_key2 \
      fragdb ctune_off_lb2 ctune_off_cap hwcvt \
    --cases engine/attn/sweep/cases_reduced_r2.json \
    --build-only --out "$BUILD"
```

Exit0 and every requested build's returncode0 are required. The bundle includes
all17 cases, binaries, immutable compiler/source/dependency/binary records,
run.py and merge.py. Copy the **entire pristine build bundle**, before any run,
to a fresh run directory on each of4 booked GB10 boxes. No source checkout,
host compiler or nvcc is needed for run-only; matching CUDA driver/runtime and
binary host libraries still are. No automatic SSH/container/service operation
is performed by the driver.

## Four disjoint case shards

On each box choose its assigned index0,1,2,3 (same kernel timing options on all
boxes). The17 cases partition5/4/4/4; all candidates of one case stay together.

```sh
RUN=/root/attn-sweep/adaptive-reduced-run0-COMMIT
flock /tmp/attn-sweep-gb10.lock python3 "$RUN/run.py" \
  --arch 121a --run-only --out "$RUN" \
  --adaptive-ci 1.0 --shard 0/4
# The other copies use1/4,2/4,3/4. Do not reuse a completed run directory.
```

`--adaptive-ci` uses min3/max6 process rounds per candidate/case; `--rounds`
controls fixed mode only. An out-of-band observed off/off null prevents early
stop and remains labelled at the cap. Kernel measurement defaults are unchanged
(warmup3, reps5, iters20, pools3, flush0, amortize1); the owner may choose another
COMMON configuration explicitly. Low-memory attempts are recorded/retried, never
silently dropped or resized. Only completed cases can produce valid totals.

## Merge on dev, with source evidence preserved

Copy complete finished run directories into reports, retaining original file
contents; then run with the same committed merge tool:

```sh
python3 engine/attn/sweep/merge.py \
  --out /home/reggie/legions/tuning-2/reports/attn-adaptive-MERGED-NEW \
  /path/to/run0 /path/to/run1 /path/to/run2 /path/to/run3
```

The output directory must not exist. Merge verifies identities, settings, all
shard indices and completed round cohorts. It archives each complete source run
under `shards/i`, writes its hash manifest, and reconstructs one summary.csv and
summary.md from raw observations. Failures remain failures. Disjoint boxes are
NOT extra replicates for a case; cross-host clocks/load still require inspection.

The committed list is the union17:16 CQ-r2 chunks for128K and all4 for32K,
including32K's distinct n8191_p24576 final chunk. The displayed
**reduced set:90.2698% of32K /90.4781% of128K off weight** is archived historical
OFF-time coverage, explicitly NOT raw call percentage, current captured-regime
coverage or full TTFT. Omitted work is not extrapolated. Old baseline means,
full denominators and source CSV SHA256 are carried by the case JSON.
