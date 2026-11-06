# DeepSeek-V4.1-Flash

Four DGX Spark (GB10) boxes, TP=4 only. Weights: [kindlingai/deepseek-v4.1-flash-exlr8-native](https://huggingface.co/kindlingai/deepseek-v4.1-flash-exlr8-native),
the official [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) weights, byte for
byte, in the exlr8 file layout: 510.6 GB, fetched whole to every box.

gr8 serves it as `ds41`. It drafts with the checkpoint's own MTP layers instead of DFlash2, and its front end speaks
the DeepSeek dialect (DSML tool calls, numeric reasoning effort). It serves text by default. Image input is opt-in
([DS41-VISION.md](DS41-VISION.md)).

## Results

Exact output (none of the speed-for-exactness switches in [FIRST-BOOT.md](FIRST-BOOT.md#3-deepseek-v41-flash-ds41-speed-against-exactness)),
Engram on, 5 chat slots, four GB10 boxes on one 200G switch.

| | tok/s |
|---|---|
| decode, one stream: code / prose / structured | 104.6-105.9 / 61.7-62.0 / 126.2-126.9 |
| prefill, cold: 8K / 32K / 128K prompt | 3,197-3,700 / 3,721-4,001 / 3,741-3,815 |
| aggregate decode: 1 / 2 / 4 / 8 streams | 88.9-91.1 / 110.0-113.6 / 119.7-131.2 / 137.8-145.9 |

`bench/bench.py` (gr8's own benchmark, `reasoning_effort` low, timed stages at uncapped GPU clocks), two runs on one
boot of `main` at 40a91f5 with the launcher's defaults, on 2026-11-02. A range is the two runs. The first word after a
128K prompt came in 34.3-35.1 s. Both runs passed smoke (11/11) and gave the same temperature-0 replies.

- Quality: 73 of 75 in each of three runs of the 75-item check of [FIRST-BOOT.md](FIRST-BOOT.md#3-deepseek-v41-flash-ds41-speed-against-exactness)
  (code, math, JSON, reasoning), exact output, on `main` at b10aae4 (2026-11-03).
- Faster, not exact: with the Q3 switches of [FIRST-BOOT.md](FIRST-BOOT.md#q3-faster-writing-for-a-small-quality-loss),
  one chat writes 162-164 tok/s on short code prompts and 8 chats write 249-254 tok/s together, and the check gives 71
  of 75 ([FIRST-BOOT.md](FIRST-BOOT.md#q1-how-many-chats-at-once) has the table).
- Our boxes run with their CPUs kept out of deep idle (PM QoS `cpu_dma_latency` 0) and with memory compacted before
  each boot. Both are box settings outside gr8.
- These boots went through our own test harness, not `./gr8 ds41 up` (see [Known issues](#known-issues)).

## Start it

1. Fetch the whole release to local NVMe on every box (510.6 GB a box).
2. Write the Engram token map once.
3. `./gr8 ds41 build` on every box.
4. `cluster.env` on every box, then `./gr8 ds41 check`, `./gr8 ds41 up` and, on the head, `./gr8 ds41 smoke`.

[DEEPSEEK-BOOT.md](DEEPSEEK-BOOT.md) has each step in order, with the disk and Engram details.

## Settings

The launcher's defaults: 5 chats at once, 262,144 tokens a chat, exact output. A request that names no reasoning effort
gets low (50), one that names no `max_tokens` gets 32,768 (`DEFAULT_MAX_TOKENS`), and one that names no `top_p` samples
with 0.95.

[FIRST-BOOT.md](FIRST-BOOT.md#2-deepseek-v41-flash-ds41-how-many-chats-and-how-long) has the measured numbers for each
choice: how many chats at once (up to 16), how long a chat can be (up to 1,048,576 tokens), faster writing for a small
quality loss, a smaller output head, the first word sooner with several chats, and long prompts while other chats write.
Each one is a config line or an `EXTRA_ENV` word, with no rebuild.

## Known issues

[KNOWN-ISSUES.md](../KNOWN-ISSUES.md#deepseek-v41-flash) has the open ones: the `./gr8 ds41` route not yet run from a
clean machine, no KV store below the device, no JSON-schema masking, recall at 1M tokens, and image requests that hold
the other chats.
