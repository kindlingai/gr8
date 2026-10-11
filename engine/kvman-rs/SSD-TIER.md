# An SSD tier for retained runs: proof of concept

Status: measured and simulated only. Nothing in the engine calls `src/tier.rs`, and it has no C entry points.

## What it would fix

With prefix reuse on, a retired session's run stays in the pool, and the session's next request skips the prefill of
the prefix it shares. A reservation that needs those pages evicts whole retained runs, the least recently retired
first. On a busy pool the run is often gone by the time the session comes back, and the request prefills its whole
prompt again: at 1,300 tok/s a 40,000-token prompt is about 31 s. The tier writes an evicted run to the box's NVMe and
reads it back when a later request continues it.

## Sizes

From the engine's caches (`engine/attn/bind.cu`, `attn.cu` `KV_BYTES` / `IDX_BYTES`, `engine/runtime/rank.cuh`
`cp_full_index`):

| unit | contents | bytes | on disk |
|---|---|---|---|
| page block | one 64-row page of every layer: 79 MLA caches (78 layers + MTP) x 64 x 656 B, 22 index caches (layers 0, 1, 2, 6, 10, .., 74 and MTP) x 64 x 132 B | 3,502,592 | 3,506,176 (856 x 4 KiB) |
| row | one token's MLA rows for one indexer group (a full-index layer and the 3 layers that reuse its list): 4 x 656 B | 2,624 | 4,096 |

A 40,000-token prefix is 625 pages, 2.19 GB. Below R every rank holds the whole run, so every rank would spill and
restore its own copy. The DFlash2 drafter's KV is not in a page block.

## Measured: kvman-ssd-bench

One GB10 box (ASUS GX10), its internal NVMe, ext4, the filesystem 99% full, an 8 GB scratch file, O_DIRECT, std
threads with one synchronous `pread`/`pwrite` each.

Page blocks, GB/s, every pass moves the whole 8 GB file. Second run, on a file the first run had written:

| threads | write, slot order | write, shuffled | read, slot order | read, shuffled |
|---|---|---|---|---|
| 1 | 5.41 | 5.40 | 5.13 | 5.08 |
| 2 | 5.88 | 5.74 | 6.96 | 7.18 |
| 4 | 5.68 | 5.74 | 7.28 | 7.31 |
| 8 | 5.74 | 5.73 | 7.18 | 7.27 |
| 16 | 5.75 | 5.75 | 6.99 | 6.88 |

The first run read the same (5.1-7.4) but three of its ten write passes ran at 1.08-1.83 GB/s, and filling the newly
created file ran at 4.09 GB/s once and 0.56 GB/s the second time. A real store would allocate its file once and
overwrite it.

4 KiB random reads (the row shape), one thread per read in flight:

| queue depth | IOPS | GB/s | p50 us | p90 us | p99 us | p99.9 us |
|---|---|---|---|---|---|---|
| 1 | 22,895 | 0.09 | 43.0 | 43.7 | 45.0 | 432.3 |
| 2 | 45,597 | 0.19 | 42.5 | 45.0 | 60.6 | 140.1 |
| 4 | 84,945 | 0.35 | 44.2 | 57.1 | 79.3 | 95.7 |
| 8 | 143,182 | 0.59 | 47.3 | 77.0 | 141.3 | 190.6 |
| 16 | 194,016 | 0.79 | 64.4 | 165.5 | 244.2 | 313.3 |
| 32 | 255,815 | 1.05 | 96.1 | 249.2 | 448.8 | 654.5 |
| 64 | 485,236 | 1.99 | 121.7 | 187.4 | 315.6 | 417.4 |
| 128 | 832,555 | 3.41 | 134.8 | 237.3 | 375.7 | 506.5 |

64 KiB random reads reach 6.87 GB/s at depth 128 (104,804 IOPS, p50 1.14 ms).

So a 2.19 GB run takes about 0.38 s to write at 5.7 GB/s and 0.31 s to read at 7.0 GB/s, against about 31 s to
prefill it again.

## Simulated: kvman-sim

`sim/traces/before.csv` and `after.csv` are two captures of four opencode sessions each (53 and 54 requests with
tools), converted by `sim/capture_trace.py`. Every request asks for `max_tokens` 32,000, so prompts past about 33,500
tokens reserve more than R (65,536) and run exclusive, which evicts every retained run.

The simulator runs the engine's pool code (`src/pool.rs`, prefix-reuse rules strict off, NOEND on, grid 4,096) with
4,608 logical and 1,920 physical pages, 8 sessions, guard 14. Time model: one prefill at a time at 1,300 tok/s (new
tokens only), no decode during a prefill, decode 30 tok/s alone, 51.5 total at 2 sessions, 64.4 at 4. The ssd policy
writes evicted runs at 5.7 GB/s before the admitting request's prefill and reads a hit's adopted pages at 7.0 GB/s.
A store of 256 GB per rank.

`lost` counts requests that shared at least one grid cell (4,096 tokens) with the session's previous request and
prefilled from 0 anyway. SSD traffic is per rank.

| trace | policy | TTFT p50 s | TTFT p90 s | wall s | prefilled tokens | avoided | pool hits | SSD hits | lost | SSD written GB | SSD read GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| before | none | 42.6 | 109.0 | 3,595 | 1,419,600 | 0 | 0 | 0 | 49 | 0 | 0 |
| before | today | 42.6 | 109.0 | 3,138 | 825,680 | 593,920 | 20 | 0 | 29 | 0 | 0 |
| before | ssd | 16.9 | 79.2 | 2,741 | 285,008 | 1,134,592 | 21 | 27 | 1 | 92.6 | 28.7 |
| after | none | 121.1 | 375.1 | 3,271 | 1,802,297 | 0 | 0 | 0 | 50 | 0 | 0 |
| after | today | 121.1 | 375.1 | 3,186 | 1,691,705 | 110,592 | 2 | 0 | 48 | 0 | 0 |
| after | ssd | 38.5 | 308.3 | 2,150 | 290,873 | 1,511,424 | 2 | 48 | 0 | 178.8 | 76.7 |

Measured TTFT in the captures, for scale: before median 58.5 s, p90 216.3 s, after median 108.7 s, p90 309.0 s. The
model has no prefill slowdown with context and no exclusive-run cost, so it is the optimistic side of the real engine.

Variations (`kvman-sim TRACE --policies ...` with the flag shown):

| trace | change | today p50 / p90 / wall | ssd p50 / p90 / wall | SSD written GB |
|---|---|---|---|---|
| before | writes at 1.1 GB/s | 42.6 / 109.0 / 3,138 | 22.7 / 85.3 / 2,809 | 92.6 |
| after | writes at 1.1 GB/s | 121.1 / 375.1 / 3,186 | 46.5 / 316.0 / 2,282 | 178.8 |
| before | `--spill retire` (every retained run written at retirement, off the critical path) | | 14.6 / 77.8 / 2,700 | 170.8 |
| after | `--spill retire` | | 35.6 / 306.5 / 2,119 | 193.5 |
| before | `--max-tokens 8192` (no exclusive runs) | 3.8 / 46.6 / 2,816 | 3.8 / 20.9 / 2,657 | 19.5 |
| after | `--max-tokens 8192` | 84.4 / 188.1 / 2,509 | 28.6 / 108.0 / 1,536 | 104.3 |
| both | `--ssd-gb 32` | | unchanged | |

A 32 GB store gives the same results because each session only comes back for its latest run.

On these traces the SSD policy removes most of the repeated prefill (80-84% of prompt tokens avoided against 6-42%
today), cuts median TTFT by 2.5-3.1x and wall time by 13-33%. It writes 92.6-178.8 GB per rank in 36-46 minutes of
replay, about 120-300 GB per hour, which matters for the drive's write endurance.

### Sparse row cache (estimate only)

`--rowcache W`: per decoded token, each of 21 indexer groups selects 2,048 tokens. Rows outside a W-token resident
window come from SSD, one 4 KiB read each. The estimate assumes the selection is spread evenly over the context.

| trace | window 16,384 | window 32,768 | window 49,152 |
|---|---|---|---|
| before | 18,802 reads (77.0 MB), 22.6 ms per token | 1,817 reads, 2.2 ms | 0 |
| after | 21,807 reads (89.3 MB), 26.2 ms per token | 8,025 reads, 9.6 ms | 1,347 reads, 1.6 ms |

Milliseconds at 832,555 IOPS (depth 128). A decode step at 30 tok/s is about 33 ms, so a 16k window would cost most of a
step on these contexts, a 32k window 7-29% of one, and a 48k window at most 5%.

## What hooking it up would take

- **Where.** `RunPool::fill` and the exclusive branch of `reserve` / `take` already hold the evicted runs (`gone`).
  The spill goes there, at the event fence, before the new grant's pages are written. The restore goes next to
  `match`: the head in `Service::event` (after a pool miss, match the stored runs with the same `begin` rule) and each
  worker in the bind hook, as it repeats `match` / `take` today.
- **Lock-step.** Store decisions must be a function of calls every rank makes in the same order: the store keyed by
  retained age, slots lowest free first, and its own drop order. The grant gains a field naming the restored run, the
  frame version moves, and the pool digest folds in the store (stored ages and slots), so a rank whose store differs
  fails at the next event as a diverging pool does now.
- **Device copies.** Each layer's binding has its own storage (`bind.cu` `create`), so one page block is a gather of 101
  pieces (79 x 41,984 B and 22 x 8,448 B) into a pinned staging buffer, and a restore is the scatter back. The MTP layer
  lives in the draft model's bindings. The spill must finish its device-to-host copy before the grant's prefill
  overwrites the pages. The disk write can trail behind as long as the store records the run as unreadable until it
  lands.
- **The rest of a run.** The DFlash2 drafter's KV pool is not in a page block, so a restored run needs it rebuilt or
  stored too. Exclusive runs past R hold different pages on each rank (the CP map), so their blocks differ per rank.
- **Wear and space.** At the rates above a box writes 120-300 GB per hour of busy agent traffic. A preallocated file
  avoids the slow first fill seen in the bench.
