# DeepSeek-V4.1-Flash

Four DGX Spark (GB10) boxes, TP=4 only. Weights: [kindlingai/deepseek-v4.1-flash-exlr8-native](https://huggingface.co/kindlingai/deepseek-v4.1-flash-exlr8-native),
the official [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) weights, byte for
byte, in the exlr8 file layout: 510.6 GB, fetched whole to every box.

gr8 serves it as `ds41`. It drafts with the checkpoint's own MTP layers instead of DFlash2, and its front end speaks
the DeepSeek dialect (DSML tool calls, numeric reasoning effort). It serves text by default. Image input is opt-in
([DS41-VISION.md](DS41-VISION.md)).

## Results

Fast mode, the default of the Docker recipe
[deepseek-v4.1-flash-exlr8-4x-spark](https://github.com/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark) (`TUNE=speed`):
gr8 `main` at [`0015fd0`](https://github.com/kindlingai/gr8/commit/0015fd04512ebc38e369475a1ce7dee76d445d77), engine
built with `DS41_MX=48`, Engram on, 32 chat slots, `CAP` 327,680, the recipe's release settings (its `knobs.sh`). Two
sets of four GB10 boxes, each set cabled in a ring, on 2026-11-08: set A for prefill and one-stream decode, set B for
many streams. Fast mode's output is not bit-identical to the exact model's: the same prompt can get a different reply.

Prefill, cold prompt (set A):

| prompt | runs | first token | prefill tok/s | decode tok/s after it, same request |
|---|---|---|---|---|
| 32K tokens | 16 | 4.12 s | 7,950 | 103.5 |
| 128K tokens | 8 | 16.6 s | 7,887 | 98.2 |
| 256K tokens | 2 | 34.7 s | 7,548 | 101.0 |

One-stream decode on short prompts (64-72 tokens, up to 512 tokens out; set A, two series of 6 runs each):

| work | code | prose | structured | counting |
|---|---|---|---|---|
| tok/s | 143.5-145.0 | 83.4-86.0 | 160.4-160.9 | 162.1-162.4 |

Many streams at once (set B, short mixed prompts, up to 512 tokens out per stream):

| streams | 8 | 16 | 32 |
|---|---|---|---|
| total decode tok/s | 200.3 | 221.3 | 226.4 |
| first token, median | 2.5-2.7 s | 6.2-6.4 s | 14.2-14.6 s |

At 32 streams the requests wait for prefill slots, so the first token comes later as the queue grows. The longest
prompt we measured with `TUNE=speed` is 262,219 tokens; `CAP` allows 327,680. `TUNE=context` (16 chat slots, `CAP`
1,048,576, not the release settings) prefilled a 1,000,660-token prompt at 2,689 tok/s (two runs, first token in
372 s).

### How we measure

- gr8's own benchmark (`bench/bench.py`) against the running server, with the recipe's settings. These runs used an
  image with the engine built from `0015fd0`, not yet the recipe's published image; a run on the published image
  follows.
- No tracing or per-stage timing on the measured requests.
- A number that compares two settings comes from paired runs: both settings on the same boxes, each on a fresh boot,
  in both orders.
- Every speed is tok/s. Every one-stream decode number gives its prompt length.
- A single number is the mean of the runs in its row; a range is the lowest and highest series. Many-stream totals:
  two series of two batches per level, the total of each series' tokens over its decode time, then the mean of the
  two series. One boot per set of boxes.

### Exact output

The exact model: none of the speed-for-exactness switches in [FIRST-BOOT.md](FIRST-BOOT.md#3-deepseek-v41-flash-ds41-speed-against-exactness),
Engram on, 5 chat slots, four GB10 boxes on one 200G switch. gr8's own launcher gives exact output by default.

| | tok/s |
|---|---|
| decode, one stream: code / prose / structured | 104.6-105.9 / 61.7-62.0 / 126.2-126.9 |
| prefill, cold: 8K / 32K / 128K prompt | 3,197-3,700 / 3,721-4,001 / 3,741-3,815 |
| aggregate decode: 1 / 2 / 4 / 8 streams | 88.9-91.1 / 110.0-113.6 / 119.7-131.2 / 137.8-145.9 |

`bench/bench.py` (gr8's own benchmark, `reasoning_effort` low, timed stages at uncapped GPU clocks), two runs on one
boot of serving code equivalent to `main` at 40a91f5, with the launcher's defaults, on 2026-11-02. A range is the two
runs. The first word after a 128K prompt came in 34.3-35.1 s. Both runs passed smoke (11/11) and gave the same
temperature-0 replies. The longest request on that boot was 163,830 tokens; `main` now allows 262,134 (#325).

The two tables come from different builds and boots, so they do not show how much of the difference fast mode alone
makes.

### Quality and the boxes

- Quality, on the 75-item check of [FIRST-BOOT.md](FIRST-BOOT.md#3-deepseek-v41-flash-ds41-speed-against-exactness)
  (code, math, JSON, reasoning): fast mode got 73 of 75 on each of two boots (2026-11-08); both missed the same two
  items, one of them cut off at the length limit. Exact output got 73 of 75 in each of three runs, on serving code
  equivalent to `main` at b10aae4 (2026-11-03).
- Our boxes run with their CPUs kept out of deep idle (PM QoS `cpu_dma_latency` 0) and with memory compacted before
  each boot. Both are box settings outside gr8.
- These boots went through our own test harness, not the recipe's image and not `./gr8 ds41 up` (see
  [Known issues](#known-issues)).

## Start it

The quickest way is the recipe
[deepseek-v4.1-flash-exlr8-4x-spark](https://github.com/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark): the same
Docker image on all four boxes, in fast mode. On every box, fetch the release (510.6 GB) with the image, then start
the image. Its README has the commands (`docker run` or Docker Compose), the cabling and the settings, and shows how to
get the exact model instead. We booted the recipe's image on four boxes on {{DATE}} (ready in {{BOOT_S}} s, smoke test
{{SMOKE}}, {{QEVAL}} of 75 on the 75-item check in fast mode): if a step fails, tell us.

With gr8's own launcher, which gives exact output by default:

1. Fetch the whole release to local NVMe on every box (510.6 GB a box).
2. Write the Engram token map once.
3. `./gr8 ds41 build` on every box.
4. `cluster.env` on every box, then `./gr8 ds41 check`, `./gr8 ds41 up` and, on the head, `./gr8 ds41 smoke`.

[DEEPSEEK-BOOT.md](DEEPSEEK-BOOT.md) has each step in order, with the disk and Engram details.

## Settings

The recipe boots in fast mode by default: 32 chats at once, `CAP` 327,680 (a request holds at most 327,670 tokens,
prompt and reply), and an engine built with `DS41_MX=48`: a decode step holds 48 rows instead of 32, so one step can
check 6 rows for each of 8 chats. Its settings come in three groups, each set in the recipe's `knobs.sh`:

- Three change the output: `GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1` and `GR8_DS_MOE_TOPK_DECODE=4`
  ([FIRST-BOOT.md](FIRST-BOOT.md#q3-faster-writing-for-a-small-quality-loss) Q3), and `GR8_DS41_HEAD_MX=1`
  ([Q4](FIRST-BOOT.md#q4-a-smaller-output-head)).
- Two speed up prefill with the same output bits: `GR8_DS_ATTN_TC_MF=2` and `GR8_DS_ATTN_TC_TILE=1` ([ENV.md](ENV.md)).
- The release stack: the v2 prefill path (`GR8_DS_PF_V2=1`) with 8,192-row chunks, its wire settings, the two-rail
  split, and Engram reads with io_uring. One of them, `GR8_DS_SH_TC=1`, changes the output. The recipe README's
  [What `TUNE=speed` sets](https://github.com/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark#what-tunespeed-sets) has
  one line for each. Engram's io_uring reads need the recipe's seccomp profile; without it, Engram reads with threads
  and gives the same output.

With more than 5 chats it also sets `GR8_DS_SESS_DRAFTS=1`: each chat drafts fewer tokens as more chats write, and
none when more than 24 write at once.

gr8's own launcher (`./gr8 ds41 up`) still defaults to exact output: 5 chats at once and `CAP` 262,144 (a request
holds at most 262,134 tokens).

With either, a request that names no reasoning effort gets low (50), one that names no `max_tokens` gets 32,768
(`DEFAULT_MAX_TOKENS`), and one that names no `top_p` samples with 0.95.

[FIRST-BOOT.md](FIRST-BOOT.md#2-deepseek-v41-flash-ds41-how-many-chats-and-how-long) has the measured numbers for each
choice: how many chats at once (measured at 5, 10 and 16), how long a chat can be (`CAP` up to 1,048,576, so a
request of up to 1,048,566 tokens), faster writing for a small quality loss, a smaller output head, the first word
sooner with several chats, and long prompts while other chats write. Each one is a config line or an `EXTRA_ENV`
word, with no rebuild.

## Known issues

[KNOWN-ISSUES.md](../KNOWN-ISSUES.md#deepseek-v41-flash) has the open ones: the `./gr8 ds41` route not yet run from a
clean machine, no KV store below the device, no JSON-schema masking, recall at 1M tokens, and image requests that hold
the other chats.
