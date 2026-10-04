# Rank health probe and boot admission (#29)

A rank that is slow or broken makes the whole TP6 engine slow or wrong, and the cause is hard to see after the boot.
The probe checks every rank **before** a boot and refuses it with a named reason. It only reads and measures: it changes
no clock, link, driver or power setting, and it refuses a node where a model is booted (`NODE_BUSY`). Automatic
recovery is out of scope: the probe says what is wrong, a person fixes it.

```bash
scripts/rank_health.sh [OUT_JSON]      # on the head, no model booted; exit 0 admitted, 3 refused
scripts/launch_tp6.sh                  # runs the probe first; RANK_HEALTH=0 skips it
python3 bench/gate/gate.py run ... --health $G6_HOME/out/rank_health.json     # stores the result in the run records
python3 tools/health/rank_probe.py local --proxy-cpu 19 --rail-ips 10.0.1.1,10.0.2.1    # one node, JSON
```

`scripts/build.sh` builds `bin/gpuprobe`, `scripts/stage.sh` copies it to every node. The nodes also need `ib_write_bw`
(package `perftest`) and `nvidia-smi`. `rank_probe.py` ships itself over ssh, so only `bin/gpuprobe` is staged.

## What is measured

| measurement | how |
|---|---|
| link rate per rail | `ib_write_bw` (1 MiB writes, 3 s) around a ring: rank r writes to rank r+1 on every rail, all ranks at the same time. A cable fault that reports "200G, no errors" and caps the NIC at ~13 Gb/s shows only in a transfer, so the sysfs state alone is not trusted. The TCP control connection of `ib_write_bw` always uses the peer's first (fabric) address, so a host firewall that allows only that subnet does not block the test of the other rail; the RDMA data path is the rail's own device and GID |
| port state and PCIe link | sysfs: port of the RDMA device that owns the rail address must be ACTIVE; the ConnectX PCIe link must run at its maximum speed and width |
| GPU clock and throttle reasons | `nvidia-smi` sampled while `gpuprobe` reads device memory (a clock is only meaningful under load) |
| wedge | `nvidia-smi -L` must answer within 15 s (a held RM lock hangs it). D-state processes and coredump writers are listed as context |
| device-memory read and host-to-device copy rate | `gpuprobe`; compared with the median of the ranks |
| proxy CPUs | the CPUs the fabric proxy threads use (`--proxy-cpu` and the four below it) must be online, in the CPU set, and not busy while the node is idle |

## Reasons

| code | meaning |
|---|---|
| `NODE_BUSY` | a `g6m4` process, a GPU at 20% utilization or more, or under 16 GiB free memory: a model is booted. Nothing else is measured |
| `PROBE_FAILED` | the node did not answer, a tool is missing (`gpuprobe`, `ib_write_bw`), or no RDMA device owns a rail address |
| `GPU_WEDGED` | `nvidia-smi -L` hangs or fails. Only a power cycle recovers a held RM lock |
| `GPU_CLOCK_LOW` | SM clock under load below the minimum (a power throttle sits at 500-850 MHz) |
| `GPU_HW_THROTTLE` | a hardware slowdown, thermal or power-brake throttle reason is active under load |
| `LINK_DOWN` | the port of a rail is not ACTIVE |
| `LINK_PCIE_DEGRADED` | the ConnectX PCIe link runs below its maximum speed or width |
| `LINK_SLOW` | the RDMA write rate out of this rank on a rail is below the floor, or more than 15% below the median of the same rail on all ranks. The line names the sending rank and the receiver (a slow receiver shows on the sender) |
| `HBM_SLOW` / `H2D_SLOW` | device-memory read / host-to-device copy rate more than 8% below the median of the ranks |
| `PROXY_CPU_OFFLINE` / `PROXY_CPU_BUSY` | a proxy CPU is offline or outside the CPU set / over 25% busy while the node is idle |

Rules that compare ranks with each other need at least three probed ranks. The absolute limits always apply. The ring must give one successful result (client exit code 0 and a bandwidth row) per rail on every rank; a missing, partial or failed result refuses the rank with `PROBE_FAILED`.

## Thresholds

Defaults, set in `Thresholds` in `rank_probe.py` and overridable with flags of `rank_probe.py cluster` (through
`PROBE_ARGS`): `--min-clock-mhz 1500`, `--min-link-gbps 80`, `--link-peer-tol 0.15`, `--hbm-tol 0.08`, `--h2d-tol 0.08`,
`--max-proxy-cpu-busy-pct 25`, `--min-free-gib 16`, `--busy-gpu-util-pct 20`, `--wedge-timeout-s 15`, `--load-seconds 4`.

Measured on six healthy nodes (three probes each, GPU clock capped at 2200 MHz, no engine loaded):

| quantity | healthy range, all ranks, all probes | default limit |
|---|---|---|
| SM clock under load | 2177-2190 MHz | 1500 MHz minimum |
| device-memory read | 234.4-237.8 GB/s (spread 1.4%) | 8% below the median of the ranks |
| host-to-device copy | 53.96-54.05 GB/s (spread 0.2%) | 8% below the median |
| RDMA write per rail | 98.01-98.05 Gb/s (spread 0.05%) | 80 Gb/s minimum, 15% below the median |
| probe time | 12.9-13.3 s | gate: under 60 s |

The healthy spread is far inside every limit, so the limits are kept. A rank with the clock cap lowered to 600 MHz read
565 MHz, 165 GB/s memory and 18 GB/s host copy, and was refused with `GPU_CLOCK_LOW`, `HBM_SLOW` and `H2D_SLOW`. The
tolerances could be tighter (the healthy spread is under 2%) but that is left until more boots show the spread over time.

## Result file

`--out` writes `format`, `ts`, `admitted`, `elapsed_s`, `thresholds`, per-rank measurements (`ranks`) and `refusals`
(`rank`, `host`, `code`, `detail`). `gate.py run --health` copies it into each run record under `rank_health`.

## Progress-thread isolation

`engine/comms/progress_cpus.h` keeps the registry of the CPUs the fabric proxy threads are pinned to. Just before the first
barrier, `g6m4` moves every other thread of the process off those CPUs. A proxy is recognized by its registered thread id (the proxy registers it when it starts), not by its CPU mask: another thread pinned to a proxy CPU is moved too (`G6_PROGRESS_ISOLATE=0` turns it off; see
`docs/SETTINGS.md`). A thread that cannot be moved is named in the log. The launcher runs the shim and the front end on
the other CPUs. Host test: `cd tools/health && g++ -std=c++20 -O1 -Wall -pthread -I../../engine/comms test_progress_cpus.cpp -o /tmp/t && /tmp/t`.

## Tests (no GPU)

`python3 -m unittest discover tools/health` runs the probe against fake nodes: each reason code, a healthy cluster that
must be admitted, a lowered clock cap, a rail at 13 Gb/s, a hung `nvidia-smi`, a busy node, a node that does not answer.
