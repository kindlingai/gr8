# DeepSeek-V4.1-Flash

Four DGX Spark (GB10) boxes, TP=4 only. Weights: [kindlingai/deepseek-v4.1-flash-exlr8-native](https://huggingface.co/kindlingai/deepseek-v4.1-flash-exlr8-native),
the official [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) weights, byte for
byte, in the exlr8 file layout: 510.6 GB, fetched whole to every box.

gr8 serves it as `ds41`. It drafts with the checkpoint's own MTP layers instead of DFlash2, and its front end speaks
the DeepSeek dialect (DSML tool calls, numeric reasoning effort). It serves text by default. Image input is opt-in
([DS41-VISION.md](DS41-VISION.md)).

## Results

Fast mode, the default of the Docker recipe
[deepseek-v4.1-flash-exlr8-4x-spark](https://github.com/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark): Engram on,
32 chat slots, `CAP` 327,680, the six fast-mode switches of [Settings](#settings), and an engine built with
`DS41_MX=48`. Four GB10 boxes on one switch. Fast mode's output is not bit-identical to the exact model's: the same
prompt can get a different reply.

| | tok/s |
|---|---|
| decode, one stream: code / prose / structured | 142.4-145.6 / 83.5-84.6 / 159.5-161.6 |
| prefill, cold: 8K / 32K / 128K prompt | 3,655-4,463 / 4,453-4,903 / 4,449-4,612 |
| aggregate decode: 1 / 2 / 4 / 8 streams | 105.7-109.1 / 142.5-150.4 / 161.3-175.4 / 210.7-228.3 |

`bench/bench.py` (gr8's own benchmark), four runs on two boots on 2026-11-05, two runs a boot. The engine was `main`
at [`35a8f1c`](https://github.com/kindlingai/gr8/commit/35a8f1c771991a87b0c6308006b74ce70faf0378) with the change
that became #359, built with `DS41_MX=48`, through our own test harness, not the recipe's image. A range is the four
runs. The first word after a 128K prompt came in 28.5-29.5 s, and a fact hidden in a 128K prompt was found in each
run. These boots had 32 chat slots, and a request could hold up to 327,670 tokens, prompt and reply.

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
  (code, math, JSON, reasoning): fast mode got 72 of 75, in one run on each of its two boots (2026-11-05). Exact
  output got 73 of 75 in each of three runs, on serving code equivalent to `main` at b10aae4 (2026-11-03).
- Our boxes run with their CPUs kept out of deep idle (PM QoS `cpu_dma_latency` 0) and with memory compacted before
  each boot. Both are box settings outside gr8.
- These boots went through our own test harness, not the recipe's image and not `./gr8 ds41 up` (see
  [Known issues](#known-issues)).

## Start it

The quickest way is the recipe
[deepseek-v4.1-flash-exlr8-4x-spark](https://github.com/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark): the same
Docker image on all four boxes, in fast mode. On every box, fetch the release (510.6 GB) with the image, then start
the image. Its README has the commands (`docker run` or Docker Compose), the cabling and the settings, and shows how to
get the exact model instead. We booted the recipe's image on four boxes on 2026-11-06 (ready in 104 s, smoke test 11
of 11, 72 of 75 on the 75-item check in fast mode): if a step fails, tell us.

With gr8's own launcher, which gives exact output by default:

1. Fetch the whole release to local NVMe on every box (510.6 GB a box).
2. Write the Engram token map once.
3. `./gr8 ds41 build` on every box.
4. `cluster.env` on every box, then `./gr8 ds41 check`, `./gr8 ds41 up` and, on the head, `./gr8 ds41 smoke`.

[DEEPSEEK-BOOT.md](DEEPSEEK-BOOT.md) has each step in order, with the disk and Engram details.

## Settings

The recipe boots in fast mode by default: 32 chats at once, `CAP` 327,680 (a request holds at most 327,670 tokens,
prompt and reply), and an engine built with `DS41_MX=48`: a decode step holds 48 rows instead of 32, so one step can
check 6 rows for each of 8 chats. It turns on six switches. Three change the output:
`GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1` and `GR8_DS_MOE_TOPK_DECODE=4`
([FIRST-BOOT.md](FIRST-BOOT.md#q3-faster-writing-for-a-small-quality-loss) Q3), and `GR8_DS41_HEAD_MX=1`
([Q4](FIRST-BOOT.md#q4-a-smaller-output-head)). Three speed up prefill: `GR8_DS_ATTN_TC_MF=2`,
`GR8_DS_ATTN_TC_TILE=1` and `GR8_DS_MHC_ROW_SHARD=1` ([ENV.md](ENV.md)). With more than 5 chats it also sets
`GR8_DS_SESS_DRAFTS=1`: each chat drafts fewer tokens as more chats write, and none when more than 24 write at once.

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
