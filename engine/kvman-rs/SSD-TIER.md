# SSD KV store: session extensions and reserved space

The engine uses `src/tier.rs` through `src/blocks.rs` and `src/stash.rs` for `KV_STORE=ssd:PATH` (`--kv-store`,
experimental). The existing `keep=sessions` (or `all`) policy now reuses stored page blocks when a conversation
continues; no new option is needed. `persist=on` also keeps committed sessions across a restart.

```text
KV_STORE=ssd:/data/gr8kv,size=200G,persist=on,keep=sessions
```

## Write only the new pages

A hit no longer consumes the stored session. It leaves the entry in its tier, most recently used, and the new run
remembers which full pages came from it. When that run is evicted or drained at orderly shutdown, it extends the
entry in place: keep the old blocks below that boundary and write only the new pages. A continuing hit on the
session's own resident run does not spill it just because the conversation grew.

For runs without a stored tail, the stored extent is `min(valid, pf_end)` rounded down to the reuse grid. Pages
beyond it would be recomputed on the next non-replay hit. With the grid disabled, the extent is `valid`: a hit may
end inside a page, so the changed partial page is rewritten while earlier full pages are reused. A run shorter than
one grid cell is not stored. Runs with CP tails past R retain their existing store path.

This is provenance-based reuse, not token-based deduplication across unrelated runs. If the previous entry has
been evicted, the prompt branches, or the provenance does not cover it, the normal store path applies. Shared
prefixes can still leave duplicates beyond the aligned prefix boundary. If an extension cannot fit or the write
policy refuses it, the earlier entry stays available.

A hit writes no index drop. An extension syncs the old entry's drop before its slots can be reused and commits the
replacement only after its copies complete. A crash between those records can lose that session's cached extent,
not turn an incomplete replacement into a valid hit. Resident sessions are not checkpointed every turn; a crash
keeps only previously committed entries, whereas an orderly shutdown drains retained runs. The index format is
unchanged; the engine's existing build/model compatibility checks still apply.

## Reserve the SSD space before opening the cache

On Linux, opening an SSD block file now uses `fallocate` (mode 0) for its entire slot capacity, rather than merely
setting a sparse file's length. The `size=` budget is rounded down to whole blocks; those blocks are fully allocated
before the cache opens. Other files cannot claim that reserved space. Not enough space, or a filesystem that cannot
preallocate, is an open/boot error rather than a delayed write failure. Reopening also reserves the extent without
changing existing block contents. Non-Linux platforms retain the existing file-sizing behavior.

The CPU reserve test checks allocated bytes and writes/reads the last slot across reopen in the temporary directory.
A separate proof was run by hand on an isolated 16 MiB tmpfs: reserving 12 MiB caused a competing 8 MiB allocation
to fail with ENOSPC. CI cannot mount that isolated filesystem, so this is manual evidence, not a CI assertion.
That competing-file proof was measured on a dev tmpfs, not on the Spark SSD filesystem; full allocation on the
Spark filesystem was checked separately below.

## c21 dev measurement

CPU pool harness, not device I/O timing: 64-row pages, reuse grid 4,096, `min_hits=1`, no write limit, one whole-index
`Keep::All` tier, runs below R, including the quit drain. A block is 3,506,176 B; GB below means 1e9 B. "New" means
content never written before in that scenario (token history through the page's last valid row plus its next token,
the MTP dependency); write amplification is written/new. The regression also stamps rows by computing turn and
checks restores against their originating run, not just token equality.

| Scenario | Before: written / new blocks | Before: rewritten blocks | Before: written / new GB | Before amplification | Extend in place: written / rewritten blocks | After amplification |
|---|---|---|---|---|---|---|
| A: one chat, 10 turns, evicted between turns | 3,519 / 555 | 2,964 | 12.34 / 1.95 | 6.34 | 512 / 0 | 1.00 |
| A0: same chat, no other traffic | 1,983 / 555 | 1,428 | 6.95 / 1.95 | 3.57 | 512 / 0 | 1.00 |
| B: two chats, 8 turns each, evicting each other | 6,678 / 1,200 | 5,478 | 23.41 / 4.21 | 5.57 | 1,152 / 0 | 1.00 |
| C: four chats sharing a 10,000-token system prompt | 2,776 / 952 | 1,824 | 9.73 / 3.34 | 2.92 | 896 / 84 | 1.10 |
| D: six chats, store smaller than the working set | 8,360 / 2,042 | 6,318 | 29.31 / 7.16 | 4.09 | 6,400 / 4,736 | 3.85 |

A store smaller than the active sessions still churns: D writes 6,400 blocks, of which 1,664 are new and 4,736
are rewrites, for amplification 3.85. Source: c21 dev `port-write-amp.log:192,207`, emitted by
`cargo test --manifest-path engine/kvman-rs/Cargo.toml --test c21_write_amp -- --nocapture`.

The after totals store fewer distinct blocks because the trailing incomplete grid cell is no longer written.
C's remaining 84 rewrites are three copies of 28 system-prompt pages beyond the shared 8,192-row boundary. With an
8,192-token system prompt (C8), the harness checks zero rewrites. It also checks two ranks' copy lists and digests,
restart, a full store under eviction pressure, and grid-off partial-page restore contents
(`tests/c21_write_amp.rs`, `tests/c21_grid_off.rs`, `tests/stash.rs`).

## Historical c21 Spark measurement (not a main+port benchmark)

Measured on a TP6 Spark ring before this port: a five-turn roughly 99K-token chat plus four pressure requests,
identical requests in both arms. OLD was the S2 TP6 arm `c704bd1d`; NEW was c21 `f0614b1`.

| Check | OLD | NEW |
|---|---|---|
| Pool self-hit writes, turns 3 / 4 / 5 (rank 0) | 1,561 / 1,561 / 1,556 pages | 0 / 0 / 0: no rewrites |
| TTFT, turns 3 / 4 / 5 | 4.26 / 3.29 / 2.09 s | 1.30 / 1.32 / 1.30 s |
| Output text | reference | byte-exact 9/9 |
| SSD file, each of 6 ranks (NEW at boot / OLD after run) | sparse (16.4 GB allocated after the run) | fully allocated: 214,746,267,648 B per rank |
| Orderly quit | not measured | 1,536 pages, one entry per rank, about 0.5 s |

After NEW quit/restart, turn 6 restored 5.39 GB / 98,304 cached tokens with one store hit and TTFT 1.81 s (cold turn 1:
53.2 s). That resumed reply had no OLD reference, so its text exactness was not compared.

**Limits:** the chat remained resident; the pressure requests did not force its eviction. Forced/repeated eviction
and extension are tested on dev only, not on the Sparks. These are historical c21 results, not measurements of this
port on current main, and not a broad task-quality or topology-performance gate.

---

# Historical SSD tier proof of concept

The original device benchmark and simulator results below predate session extension. Their write volumes describe
the original policy, not the feature above. The engine now uses `src/tier.rs` through `src/blocks.rs` and
`src/stash.rs`; see also [REMOTE-TIER.md](REMOTE-TIER.md), Stage B.

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
| before | `--max-tokens 8192` (no exclusive runs; a request stops at 8,192 tokens) | 3.8 / 46.6 / 2,416 | 3.8 / 20.9 / 2,257 | 19.5 |
| after | `--max-tokens 8192` | 66.5 / 141.6 / 2,257 | 21.3 / 92.9 / 1,335 | 103.0 |
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

### Growth on demand next to the SSD tier

Since the pool lets runs grow on demand (`Pool::grow`, `Driver::step`), kvman-sim also models that: a run below R
holds its prompt + 1 + guard rows at admission and takes a page each time its tokens reach the end of its pages. A
request is admitted, and a run grows, only while every live run can still reach prompt + `max_tokens` + guard (the
safe state). Otherwise the request waits in the queue or the session waits for a page. `live` is the sessions resident
on average, `waited s` the session-seconds spent waiting for a page.

| trace | policy | TTFT p50 s | TTFT p90 s | wall s | prefilled tokens | avoided | pool hits | SSD hits | lost | SSD written GB | SSD read GB | live | waited s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before | today | 42.6 | 109.0 | 3,138 | 825,680 | 593,920 | 20 | 0 | 29 | 0 | 0 | 1.25 | 0 |
| before | ssd | 16.9 | 79.2 | 2,741 | 285,008 | 1,134,592 | 21 | 27 | 1 | 92.6 | 28.7 | 1.14 | 0 |
| before | grow | 4.1 | 25.3 | 2,678 | 285,008 | 1,134,592 | 48 | 0 | 1 | 0 | 0 | 1.34 | 0 |
| before | grow+ssd | 4.1 | 25.3 | 2,679 | 285,008 | 1,134,592 | 48 | 0 | 1 | 6.4 | 0 | 1.34 | 0 |
| after | today | 121.1 | 375.1 | 3,186 | 1,691,705 | 110,592 | 2 | 0 | 48 | 0 | 0 | 1.23 | 0 |
| after | ssd | 38.5 | 308.3 | 2,150 | 290,873 | 1,511,424 | 2 | 48 | 0 | 178.8 | 76.7 | 1.23 | 0 |
| after | grow | 114.1 | 389.1 | 2,988 | 1,388,601 | 413,696 | 12 | 0 | 38 | 0 | 0 | 1.32 | 0 |
| after | grow+ssd | 32.5 | 303.3 | 2,208 | 319,545 | 1,482,752 | 11 | 38 | 1 | 107.6 | 61.0 | 1.31 | 0 |

On the first trace growth alone keeps nearly every retained run (48 pool hits against 20): median TTFT 42.6 s ->
4.1 s, p90 109.0 s -> 25.3 s, wall time -15%. On the second, 25 of the 54 requests are past R with their `max_tokens`
(9 of 53 in the first). Those still take the whole pool at once and evict every retained run, so growth gains 6% of
wall time and the SSD tier does most of the work. With `--max-tokens 8192`, growth gives 2.8 / 7.0 s TTFT (p50 / p90)
and 2,239 s on the first trace against today's 3.8 / 46.6 s and 2,416 s, and 78.7 / 164.9 s and 2,063 s on the second
against 66.5 / 141.6 s and 2,257 s. No session waited for a page in any of these runs: the safe-state check at
admission keeps enough pages for every admitted session, so the cost of safety is a later admission, not a stall.

## What hooking it up would take

- **Where.** `RunPool::fill` (from `reserve`, `take` and `grow`) and the exclusive branch of `reserve` / `take` already
  hold the evicted runs (`gone`).
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
