# Changelog

## 2026-10-04: rank health probe, boot admission, progress-thread isolation (#29)

- New `tools/health/`: `rank_probe.py` probes every rank before a boot (RDMA write rate per rail around a ring of the ranks, GPU clock and throttle reasons under load, a driver wedge check, device-memory read and host-to-device copy rates compared between ranks, proxy CPU state) and names the reason for each refusal. `gpuprobe.cu` is its GPU part (built by `scripts/build.sh`, copied by `scripts/stage.sh`). The probe only reads and measures, and refuses a node where a model is booted.
- `scripts/launch_tp6.sh` runs the probe first and refuses to boot (exit 3) when a rank fails; `RANK_HEALTH=0` skips it. The shim and front end now run off the proxy CPUs.
- Engine: `G6_PROGRESS_ISOLATE=1` (default) moves non-proxy threads off the proxy CPUs before the first barrier. The thresholds were checked on six healthy nodes (`tools/health/README.md`).
- `bench/gate/gate.py run` needs `--health FILE` (or `--no-health`) and stores the probe result in every run record; `baseline` and `check` refuse records without an admitted probe result. A missing or failed RDMA write result refuses the rank. Isolation exempts only the registered proxy thread ids.

## 2026-10-04: pair-owner QO prefill at TP4 (`G6_PF_OWN=4`)

- **`G6_PF_OWN=4`** (TP4 builds only; [docs/DESIGN-QO-PAIR.md](docs/DESIGN-QO-PAIR.md)). OWN=3's attention block split
  over the head pairs {0,1} and {2,3}: each rank runs its pair's 32 heads on its own owner rows and its counterpart's
  (rank ^ 2), sends the counterpart's o_proj partial to that rank alone, and sums the two partials in a fixed order.
  The pair weights roll through two slots per layer, built from the rank pack and the pair partner's half over RDMA
  (`G6_PF_QP_ROLL`, a new bulk fabric on port + 56), so the cost is about 1.3-1.4 GB of device memory plus 84 MB pinned per rank instead of the 13 GB QO pack (about
  1.4 KB per position of it scales with `G6_PF_QP_CTX`).
  `tools/convert/pairpack.py` builds resident pair packs (`--qp-pack`) for smaller pools. Ported from adapt's TP4 tree.
- **`launch_tp4.sh` runs `G6_PF_OWN=4` by default** (`QP_ROLL=2`, `QP_CTX=$CAP`, `QP_EARLY=1`). At 2200 MHz: 8K/32K
  prefill 1,318/1,288 tok/s (OWN=1: 816/654), 87K TTFT 73.3 s (156.9 s), decode unchanged within noise, lowest
  MemAvailable 5.2 GiB at a 393,216 pool with a 250K prompt.
- Fabric: reduce-scatter collectives can go to one peer (`Fabric::Rs::p2p`), and bulk jobs can address one peer
  (`BulkJob::peer`). Defaults keep the old behaviour. The TP6 g6m4 and m4sim SASS is identical to main.

## 2026-10-04: TP4 (four Sparks) end to end, and pinned host-to-device copies

- **TP4.** The engine builds and runs at `G6_NR=4` (`G6_NR=4 scripts/build.sh`). Every per-rank width that was a TP6
  literal (attention head groups, vocab slice, routed-expert blocks and scratch, dense channels, buffer sizes, rank
  count checks) now comes from `engine/g6nr.h`; at NR=6 the values are the old ones and the TP6 GPU code is unchanged.
  The converter takes four GPTQ dirs (`tp4/r0..r3`) and writes four rank packs (`plan.set_nr`, `nr: 4` in the
  manifests); `qopack.py` takes four rank packs; `build_packs.sh` picks TP4 from the GPTQ dir. New `launch_tp4.sh` and
  `config/cluster.env.tp4.example`; `build.sh BUNDLE_LIBS=1` and `stage.sh SHARED=1` for nodes without the CUDA
  toolkit and packs on a shared mount. TP4 runs `G6_PF_OWN=1`: the 13 GB QO pack does not fit next to a 90 GB TP4
  rank pack. First numbers and the open QO question are on #2.
- **Pinned host-to-device copies** (`engine/h2d.h`, `g6h2d::copy`). Every copy from pageable host memory to the GPU
  goes through a per-thread pinned buffer, so a 64 KiB-page kernel cannot hang on it (#6). Host code only.

## 2026-10-04: gate harness (`bench/gate/`)

- New: `bench/gate/gate.py` measures a build (KL drift, needle, long prompt, pbench, workloads, drafter counters), appends one JSON line per run, builds a baseline per topology and checks a build against it. See `bench/gate/README.md`. No engine change.

## 2026-10-04: S2 multi-session snapshot (since b85a7a6, the K1H2 snapshot)

The default build (`scripts/build.sh`) is still K1H2. The new multi-session code is compiled in only with
`EXTRA=-DDEC_S2 scripts/build.sh`, and it is turned on at run time with `g6m4 --s2`. See KNOWN-ISSUES.md: S2 still has an open crash on long multi-chat decode.

- **S2 batched multi-session decode (opt-in, `-DDEC_S2`).** New files: `engine/runtime/s2.cuh`, `s2_batch.h`,
  `draft_shape.h`, `engine/runtime/s2/*` (scheduler, sessions, run pool, control channel, native dispatch).
  - Up to 8 sessions decode in one step. Sessions are grouped into MTP row classes (S = 3..8 at mtp 2 use ceil(S/2) groups per round).
  - Each session's KV is one contiguous run in a shared pool. A slot can move to a new base, and the 2->1 / S=1 transitions are handled.
  - Prefill can start at a base page. Below R the loop runs on the CP binding (`g6m4 --s2`).
  - The rank-0 shim talks to the native adapter over persistent control sockets.
- S2 slot views are sized for MX decode rows, not for the prefill chunk. This saves about 8.9 GB per rank at the 614K boot.
- A batched S2 chain draft step now has one row per session.
- If a step fails, S2 prints the fabric / attention / CP error words before the fail-closed check.
- New per-step timer `G6_STEP_TMR=1`, off by default. It prints a per-session request/event label for each step (TAIL / DECODE1 / ROUND / MIXED).
- Control-path disconnect logging: on a recv/send failure the engine prints ret, errno, remaining bytes and frame counts before it throws the epoch-fatal error.
- SELX (`engine/attn/selx.cuh`): our own exact top-2048 select for prefill index tiles. It runs on tiles with n >= 32 rows and context >= `G6_PF_SELX_CTX`. Its output is identical to SEL32's. Unset = SEL32 (unchanged behaviour).
- CP decode `-DCP_F1A`: the phase-2 per-partial wait + get is fused into one launch (off by default).
- The front end metrics have a hook for S2.
