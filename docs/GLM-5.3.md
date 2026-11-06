# GLM-5.3

Three to six DGX Spark (GB10) boxes, one rank per box (TP=3 to TP=6). Weights: the exlr8 release on Hugging Face,
[kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4).

## Results

| | TP=3 | TP=4 | TP=5 | TP=6 |
|---|---|---|---|---|
| experts | `bit225`, 2.25 bpw | `home`, 3.25 bpw | `home`, 3.25 bpw | `home`, 3.25 bpw |
| drafting | MTP | DFlash2 or MTP | MTP | MTP |
| decode\*, code / prose | 41.1 / 31.7 tok/s | 52.1 / 34.2 tok/s | 51.0 / 38.0 tok/s | 57.4 / 43.1 tok/s‡ |
| aggregate\*, 2 / 4 streams | 50.6 / 63.8 tok/s | 54.0 / 69.0 tok/s | 61.8 / 76.3 tok/s | 67.6 / 85.5 tok/s‡ |
| prefill, 8K prompt, cold | 635 tok/s | 1,357 tok/s | 1,554 tok/s | 1,808 tok/s‡ |
| prefill, 8K prompt, cached | 58,655 tok/s | 60,129 tok/s | 60,455 tok/s | 62,337 tok/s |
| KV pool (longest context) | 307,136 tokens (410,624 with int4§) | 485,376 tokens (513,536 with int4§) | 502,848 tokens | 614,400 tokens (857,600 with int4§) |
| requests that share decode steps | up to 32,768 tokens | up to 32,768 tokens | up to 32,768 tokens | up to 131,072 tokens |
| requests decoding at once | 8 | 8 | 8 | 8 |

\* [RigMark](https://github.com/alexellis/rigmark) 1.0.0 (`reasoning_effort` low), booted from the Hugging Face release
with `./gr8 glm53 up` and its defaults (TP=4 with `--drafter`, for DFlash2). GPU clocks uncapped (the full application
clock; the 2200 MHz cap is only for PR-to-PR gate comparisons). One run per fresh boot, the mean of two boots, on
2026-10-24. Each pair of boots gave identical temperature-0 replies, and smoke passed 14/14 on every boot. RigMark's
structured row is left out: it measures predictable output, not agent speed. TP=3 to TP=5 ran on a build of #140
before it landed, and TP=6 on 8da82ff. Six boxes on one 200G switch.
‡ One of the six boxes was capped at 2200 MHz (it powers off under sustained load at its full clock), so the TP=6
column would be a little higher on six uncapped boxes.
§ `--kv-format int4` (experimental): 4-bit KV latent, about 3% slower decode at TP=4. The TP=3 and TP=6 int4 figures
are estimates, not yet booted (TP=6 at its default `SHARE_LIMIT`, 131,072). [KV4.md](KV4.md)

[kindling.json](../kindling.json) has this table in a form a program can read. A PR that changes one changes the other.

## How it works

- One model: the [exlr8 release](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) on Hugging Face. Each box
  fetches only its own part, once, and boots from its cache in about 40 s. Each rank plans its weights from the
  release's files at load. The routed experts and the dense trellis matrices are decoded inside the kernels.
- Several requests at once: up to 8 sessions share each decode step. Each request's output is a valid continuation at
  its sampling settings, but when requests share decode steps its tokens and logprobs can differ slightly from running
  it alone.
- Speculative decoding: every round drafts with the MTP layer, or at TP=4, for a request that runs alone, with the
  DFlash2 drafter when that is expected to commit more tokens per ms (DFlash2 wins on code, MTP on prose). The draft
  count comes from a live model of step cost and acceptance.
- Prefix reuse: a request that continues an earlier conversation re-prefills only the new part. A continued 20K
  conversation takes 7.1 s instead of 17.3 s at TP=4.
- Opt-in runtime steering: one of up to eight validated rank-one profiles can be selected per request across decode, prefill and target
  verification, with steering identity included in prefix reuse. The fused operator adds no CUDA launches to the
  production path. See [STEERING.md](STEERING.md).
- Prefill attention split by query and head owner across ranks ([DESIGN-QO-PAIR.md](DESIGN-QO-PAIR.md)).
- Context-parallel KV: the first `SHARE_LIMIT` tokens (32,768 at TP=4, 65,536 at TP=3) are on every rank, later pages (their
  latent and their indexer keys) are spread across ranks, so the pools above fit. Requests up to that size share decode steps.

## What you need

Every box needs what the README's [What you need](../README.md#what-you-need) lists. For GLM-5.3, also:

- Three to six DGX Spark or other GB10 boxes (sm_121a, 128 GB unified memory), one rank per box, with nothing
  else holding much memory. A rank's weights take about 83 GB at TP=4 and 78 GB at TP=3. The KV pool and the step
  graphs need most of the rest.
- Disk per box: about 90 GB at TP=4 (its experts, the 11 GB of dense and vocab files, the drafter) and about 85 GB at
  TP=3. One cache holds both, because their parts share the sparse expert files.

## Start it

The model is `glm53`, and [DEPLOY.md](DEPLOY.md) is the guide: the model's release, how the boxes find each other, then
one short section for each way to run it, each ending with a passing `smoke`: [bare metal](DEPLOY.md#bare-metal),
[Docker Compose](DEPLOY.md#docker-compose) and [Kubernetes](DEPLOY.md#kubernetes). An agent that starts gr8 for
someone asks them the questions in [FIRST-BOOT.md](FIRST-BOOT.md) first.

## The model cache

`--model https://huggingface.co/<org>/<repo>[@rev]` fetches into `--model-cache` only what this box's rank needs: its
expert halves (sparse files, with every half checked against its xxhash64), its index slices, the manifests, the dense
and vocab files and the tokenizer. A fetch resumes where it stopped. A boot uses the commit recorded at the last fetch,
so it works offline once the cache is full, and boxes on different commits refuse to boot together. `HF_TOKEN` is sent
only when it is set. `--model DIR` boots from a local copy of the release instead.

`--variant` picks the expert widths:

| variant | routed experts | default at |
|---|---|---|
| `home` | 3.25 bpw: in each layer, the 64 experts with the highest K3 error at K4 and the rest at K3 | TP=4, TP=6 |
| `bit225` | about 2.25 bpw, down1's size: down projections at K3 in the layers where bits lower KL the most, the rest at K2 | TP=3 |
| `bit325` | 3.25 bpw, home's size: down projections at K4 in the layers where bits lower KL the most, the rest at K3 (KL 1.5-7% lower than home, decode ~2.5% slower) | |
| `bit243`, `bit260`, `bit338` | about 2.43, 2.60 and 3.38 bpw, built the same way | |
| `down1` | about 2.25 bpw | |
| `up1` | about 4.0 bpw (lower `--cap`) | |

The release's model card has each variant's size and its KL against bf16. A release from before the `bit` variants
boots `down1` at TP=3. `home` and `bit325` do not fit at TP=3: they put 108.7 GB of weights on
each rank. A boot that would run a box out of memory stops
while it allocates the KV pool, and names `--cap` and `--variant`.

## Options worth knowing

![KV tiers](kv-tiers.svg)

The first `SHARE_LIMIT` tokens of a session are on every rank and later pages are spread across the ranks; below that
pool, block stores keep evicted prefixes. With `--kv-rows` the rows past `SHARE_LIMIT` can also live in a rows store,
read back as decode selects them ([Remote KV](#remote-kv)).

- `--cap` is the KV pool in tokens, and so the longest context. Lower it first when a box runs short of memory.
- `--sess` is how many requests decode at once (1-8). Each one adds step graphs: about 2.5 GiB at 8.
- `--reuse 0` turns prefix reuse off.
- `--kv-store` and `--kv-rows` (experimental, off by default) put KV below the pool, on an SSD or a KV node: see
  [Remote KV](#remote-kv) below.
- `--kv-format int4` (experimental) caches the KV latent in 4 bits: about 1.34x the context at TP=4 (CAP 513,536 against
  408,832; the fp8 default has since grown to 485,376 and int4 is not re-measured) for about 3% slower decode, with needles found as with the default fp8 row ([KV4.md](KV4.md)).
- `--dflash 0` runs MTP only at TP=4. `--mtp` is the MTP draft count (2).
- `--steering-profile DIR[,DIR...]` loads local schema-v1 profiles; with an HF model, `hf:default` or `hf:NAME`
  explicitly selects a bundled profile. Without the flag, bundled profiles are ignored. Requests select a loaded profile
  by name with `steering_profile` and `steering_scale`; `--steering-force SCALE` (one profile) is the process-wide
  qualification mode.
- `--extra-env "NAME=value ..."` passes engine settings to every rank ([SETTINGS.md](SETTINGS.md)). The
  boxes compare them, and a `G6_*` variable in the environment is refused for that reason.

## Remote KV

A KV node is a box with spare memory that serves it to the cluster over RDMA: `./gr8 kv up --gb N` on that box, and
`remote:auto` in the options below finds it through the mentat group. The cluster can use it in two ways. Both are
experimental and off by default.

| | block stores (`--kv-store`) | rows store (`--kv-rows`) |
|---|---|---|
| holds | sessions and prefixes the pool evicted | a live session's rows past `SHARE_LIMIT` |
| gives | a later turn or a shared prefix reads its KV back instead of prefilling it | with `local=`, one request can run past what local memory holds |
| longest context | unchanged (`--cap`) | about 600K tokens at TP=4 (`--cap 655360`) |
| stores | `ssd:PATH`, `remote:auto`, `remote:HOST:PORT` | `remote:auto`, `remote:HOST:PORT`, `mem:` |

Block stores alone do not extend a single request: a request still needs local pages for all of its context. They
keep the KV of finished turns, so an agent's next turn after its session left the pool costs a read instead of a
prefill. `--kv-keep` (or `keep=` per store) picks what they keep: `system` prompts, `sessions`, or `all`.

The rows store does extend it. Each session's first `SHARE_LIMIT` tokens stay on every rank. The rows past that are
spread across the ranks, and the rows store puts each rank's share on the KV node as its pages fill. With `local=PAGES`
a rank keeps only the last PAGES pages of each long session's tail and gives the rest back to the pool. After each
indexer selection, decode reads the rows it picked from the node. `place=PAGES` caps the pages a rank holds, so
`--cap` may pass what local memory holds. A finished session keeps its records on the node, and its next turn
prefills only the new tokens.

The TP=4 long-context setup, in `cluster.env`:

```
KV_STORE=ssd:/data/gr8kv,size=4G,persist=on,keep=system;remote:auto,size=4G,keep=all
KV_ROWS=remote:auto,size=12G,local=40,place=1600
CAP=655360
```

The SSD store sits on each box. The KV node needs at least every rank's `remote:` sizes together: here 4 x (4G + 12G) = 64G.

Measured on four GB10 boxes and a KV node on 200G RoCE:

- A 445,855-token prompt prefilled in 613 s, and the next turn answered in 11.4 s from the records the node kept.
- With `fmt=int4` (416-byte rows instead of 656), a 445,255-token prompt prefilled in 581 s and its next turn took 8.8 s.
- RigMark decode is the same with the rows store on as off.

The limits:

- With the rows store, a session past `SHARE_LIMIT` runs alone and other long requests wait for it. Without it, two or
  three such requests can share decode steps.
- Buffers that grow with `--cap` (prefill staging, attention states, drafter caches: about 2.6 GB a rank at 655,360)
  take memory that would otherwise be KV pages. At `place=1600` a 4-agent replay took 1,408-1,691 s against about
  1,060 s with the rows store off. Choose `--cap` for the longest session you need.
- Without `local=`, the rows store keeps a copy of the tail and every local page stays in use, so the longest
  context is still `--cap`.

[engine/kvman-rs/REMOTE-TIER.md](../engine/kvman-rs/REMOTE-TIER.md) covers the block stores, their transports and
measurements. `./gr8 knobs KV_STORE KV_ROWS` describes every `KV_STORE` and `KV_ROWS` field, and
[SETTINGS.md](SETTINGS.md) the engine's `GR8_RT_*` settings.

## Thinking

GLM 5.3 always thinks: its chat template has no switch to turn thinking off. A chat request controls how much it thinks
with these fields:

| field | values | on GLM 5.3 |
|---|---|---|
| `reasoning_effort` (top level, or in `chat_template_kwargs`) | `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | The template names `low`, `high` and `max`. Another value becomes the nearest of those, ties going higher: `minimal` is `low`, `medium` is `high`, `xhigh` is `max`. Unset is `max`. `none` and any other value are a 400 |
| `thinking_budget` | an integer, or null | While the think block is open, the front end puts `\n</think>\n\n` in place of the reply's Nth token (so N-1 reasoning tokens come first), and the answer follows. 0, null or unset takes the server default, as in TensorFold. A negative number is no limit |
| `chat_template_kwargs.enable_thinking` | `true`; `false` with `--allow-thinking-off 1` | `false` is a 400 that names `reasoning_effort` `low`, `thinking_budget` and `--allow-thinking-off` instead |
| `response_format` | `{"type": "text"}`, `json_object`, `json_schema` (chat) | Without tools the engine masks the reply to the schema's XGrammar grammar (`GRAMMAR=1`), and the front end checks it after generation: the reply's first JSON value, and for `json_schema` the schema ([frontend/README.md](../frontend/README.md)) |

`usage.completion_tokens_details.reasoning_tokens` counts the reply's tokens through its `</think>`.
`--default-reasoning-effort` and `--default-thinking-budget` set the defaults for requests that name none (`./gr8 knobs DEFAULT_`).
The engine puts the close in place of the token itself (an insert rule, PROTOCOL.md), so a cut costs one more main step.
With an engine from before that, the front end ends the run and continues from the prompt, the reply and the close,
which prefills the prompt's last partial 4,096-token cell and the reasoning again: 2.5 to 4.2 s at TP=4's cold prefill
rate (frontend/README.md). A request that uses `logprobs` cannot have a budget.

## Troubleshooting

The README's [Troubleshooting](../README.md#troubleshooting) covers the boot. On GLM-5.3:

- Decode slows past `SHARE_LIMIT` tokens of context: from there the KV is spread across ranks, and every step gathers from
  all of them.
- A request past `SHARE_LIMIT` holds its first `SHARE_LIMIT` tokens on every rank, so it needs that many pages free on
  each. Its size is the prompt plus `max_tokens`; a chat request without `max_tokens` gets what keeps it within the
  limit, but at least 8,192 tokens, so a prompt past about 57,000 tokens is such a request. Several of them share
  decode steps when the pool has room for each one's replicated prefix: at TP=4's default `SHARE_LIMIT` (32,768 tokens,
  512 of a rank's 2,280 pages each) two or three stay resident together; `--share-limit 65536` replicates more per
  session (one resident, faster 32K-64K-context decode) at a shorter longest context (387,072 at TP=4) and shares the replicated
  prefix only up to 32,768 tokens.
- A request with `prompt_logprobs` runs its prompt as decode steps (about 13 ms a token at TP=4) and holds the others
  meanwhile. The shim caps those prompts at 1,024 tokens.
- `top_k` and `min_p` are ignored with a warning. `logprobs` and `prompt_logprobs` work up to k = 20.
  [frontend/README.md](../frontend/README.md) has the full API.

## Known issues

Open bugs on GLM-5.3: [KNOWN-ISSUES.md](../KNOWN-ISSUES.md#glm-53).
