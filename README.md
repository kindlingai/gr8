# gr8: a super-fast local AI engine

Need help? Join us on discord: https://discord.gg/M7XTrRJW3

Full GLM-5.3 on DGX Spark (GB10). *Beta release. If you run it, please tell us what breaks: issues and PRs are welcome.*

gr8 is a from-scratch CUDA inference engine for the full GLM 5.3 model (MoE with 256 experts, MLA attention with the
sparse indexer, one MTP layer) on three to six DGX Spark (GB10) boxes, tensor-parallel across all of them. It
uses no vLLM, PyTorch or NCCL at serve time. Each box runs one `g6m4` process, the ranks talk over RoCE v2 through
their own verbs fabric, and an OpenAI-compatible API runs on the first box.

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
are estimates, not yet booted (TP=6 at its default `SHARE_LIMIT`, 131,072). [docs/KV4.md](docs/KV4.md)

[kindling.json](kindling.json) has this table in a form a program can read. A PR that changes one changes the other.

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
- Opt-in runtime steering: one validated rank-one profile can be selected per request across decode, prefill and target
  verification, with steering identity included in prefix reuse. The fused operator adds no CUDA launches to the
  production path. See [docs/STEERING.md](docs/STEERING.md).
- Prefill attention split by query and head owner across ranks ([docs/DESIGN-QO-PAIR.md](docs/DESIGN-QO-PAIR.md)).
- Context-parallel KV: the first `SHARE_LIMIT` tokens (32,768 at TP=4, 65,536 at TP=3) are on every rank, later pages (their
  latent and their indexer keys) are spread across ranks, so the pools above fit. Requests up to that size share decode steps.

## What you need

- Three to six DGX Spark or other GB10 boxes (sm_121a, 128 GB unified memory), one rank per box, with nothing
  else holding much memory. A rank's weights take about 83 GB at TP=4 and 78 GB at TP=3. The KV pool and the step
  graphs need most of the rest.
- A RoCE v2 fabric with two rails: each box's ConnectX-7 has one address on each PCIe root. We use one 200G switch.
  At TP=3 or 4, `--shape ring` runs without one: each box's f0 port is cabled to the next box's f1 port, and at TP=4
  the diagonal's writes are relayed by the box between, through a 32 MB ring per fabric (about 0.25 GB pinned per
  rank). It costs about 10% of single-stream decode (4-5% at 4 streams). memlock must be unlimited. With mentat
  (`--mentat localhost --tp 3 --shape ring`, mentatd 0.24.0 or later on every box), the cabling can be any ring: the
  boxes claim a ring from mentat and take their rank order and RDMA devices from its one answer, so a scheduler such as
  a Kubernetes LeaderWorkerSet can place them in any order. Tag both addresses of each ConnectX port `rdma` in mentat
  (`MENTAT_ANNOUNCE_IFACES`).
- A LAN between the boxes with TCP ports 7340 and 7341 open. The boxes need no ssh between them: the head engine
  passes each request to the other ranks over its control links.
- Disk per box: about 90 GB at TP=4 (its experts, the 11 GB of dense and vocab files, the drafter) and about 85 GB at
  TP=3. One cache holds both, because their parts share the sparse expert files.
- Proactive memory compaction off on every box ([docs/DEPLOY.md](docs/DEPLOY.md#before-you-start) has the commands).
  A rank fills the box to about 2 GB under its limit, and at that level the kernel's background compaction moves pages
  under the engine in bursts that delay every collective.
- To build: CUDA 13 (`nvcc` with `sm_121a`), g++ with C++20, CMake, clang and libclang-dev (the front end builds XGrammar), rdma-core / libibverbs-dev, Rust through
  [rustup](https://rustup.rs) (the front end needs Rust 1.85 or later; Ubuntu 24.04's `cargo` package is 1.75), and
  Python 3 with venv. The stock DGX OS (Ubuntu 24.04) has all of them except Rust. Or, in place of all of these, Docker
  with the NVIDIA container toolkit (DGX OS has both): see [Docker Compose](docs/DEPLOY.md#docker-compose).

## Quick start

[docs/DEPLOY.md](docs/DEPLOY.md) is the guide to starting gr8. Two settings are required: the model (`glm53` or
`glm53f`, and a release of it) and how the boxes find each other (a peer list, or mentat). Then one short section for
each way to run it, each ending with a passing `smoke`: [bare metal](docs/DEPLOY.md#bare-metal),
[Docker Compose](docs/DEPLOY.md#docker-compose) and [Kubernetes](docs/DEPLOY.md#kubernetes).

### With model-loader

[model-loader](https://github.com/kindlingai/model-loader) can fetch the weights instead of gr8. The release comes
from Hugging Face once, or from a NAS that holds the master copy, and the boxes copy the pieces between them over the
LAN. Each box runs model-loader next to gr8, from model-loader's `deploy/compose` or `deploy/spark-os`, with the
release (and the drafter) in its `catalog.yaml`. Then on every box:

    cp docker/.env.example docker/.env        # MODEL, PEERS and DRAFTER as above
    docker compose -f docker/compose.yaml -f docker/model-loader.yaml pull
    docker compose -f docker/compose.yaml -f docker/model-loader.yaml up -d

`docker/model-loader.yaml` mounts model-loader's store read-only (`MODEL_LOADER_STORE`, default
`/var/lib/model-loader`), sets `FETCH=model-loader`, and starts gr8 once model-loader answers. gr8 asks model-loader
for the pieces its rank needs, waits until they are held, checks every expert half's xxhash64, and reads the weights
in place. gr8 finds the release and the drafter in model-loader's catalog by their repos, whatever the catalog
calls them. No `prepare` is needed: model-loader keeps the pieces between boots. Bare metal takes the same options:
`--fetch model-loader --model-cache /var/lib/model-loader` (KNOBS.md). Give the catalog's width files small segments
(model-loader's `deploy/compose/catalog.yaml` does), or each box holds every width file whole.

## Running it

Every box runs the same `./gr8 glm53 up` ([docs/DEPLOY.md](docs/DEPLOY.md)). The first address in `--peers` is the
head: it runs the API on port 8100. Before any engine starts, the boxes compare their options, model and engine build,
and if one box differs, every box stops and prints the difference. Without `--peers`, the boxes find each other through
a local mentat daemon (`--mentat localhost`, the default).

| command | what it does |
|---|---|
| `up [OPTIONS]` | start this box's supervisor and follow it until the cluster serves |
| `down`, `restart` | stop every box, or stop and boot again (in Docker: `down`, then `up -d` on every box) |
| `status` | every box's state, engine and free memory, and the head's front end and `/health` |
| `health` | this box alone, in one line: exit 0 when it serves (on the head, also its API's `/health`), else 1. The compose healthcheck, and a Kubernetes `exec` readiness probe on any rank |
| `logs [rank\|shim\|front\|supervisor] [-f]` | a log from any box |
| `prepare [OPTIONS]` | fill `--model-cache` with this box's part, without booting |
| `check`, `show-config` | what would stop this box's boot; every option and where it came from |
| `smoke`, `bench` | the smoke test, and prefill and decode benchmarks, against the API on this box (run them on the head) |
| `build` | the engine and the Rust front end, into `bin/` |

Options come from the command line, then a config file (`@FILE`, else `./cluster.env`; see
[config/cluster.env.example](config/cluster.env.example)), then the environment. With the options in `cluster.env`,
`./gr8 glm53 up` alone is enough. [KNOBS.md](KNOBS.md) lists every option.

When a rank's engine exits, the head stops every box: `status` names the rank and shows its last log line, and the
supervisor log names the line that says why. Logs are in `out/` on each box (`supervisor.log`, `g6m4.r<r>.err`, and
`shim.log` and `front.log` on the head). Each boot moves the previous engine logs to `out/prev1`, and keeps the three
boots before it (`prev1` is the newest), so a restart does not overwrite the boot you want to look at.

### The model cache

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

### Options worth knowing

![KV tiers](docs/kv-tiers.svg)

The first `SHARE_LIMIT` tokens of a session are on every rank and later pages are spread across the ranks; below that
pool, block stores keep evicted prefixes. With `--kv-rows` the rows past `SHARE_LIMIT` can also live in a rows store,
read back as decode selects them ([Remote KV](#remote-kv)).

- `--cap` is the KV pool in tokens, and so the longest context. Lower it first when a box runs short of memory.
- `--sess` is how many requests decode at once (1-8). Each one adds step graphs: about 2.5 GiB at 8.
- `--reuse 0` turns prefix reuse off.
- `--kv-store` and `--kv-rows` (experimental, off by default) put KV below the pool, on an SSD or a KV node: see
  [Remote KV](#remote-kv) below.
- `--kv-format int4` (experimental) caches the KV latent in 4 bits: about 1.34x the context at TP=4 (CAP 513,536 against
  408,832; the fp8 default has since grown to 485,376 and int4 is not re-measured) for about 3% slower decode, with needles found as with the default fp8 row ([docs/KV4.md](docs/KV4.md)).
- `--dflash 0` runs MTP only at TP=4. `--mtp` is the MTP draft count (2).
- `--steering-profile DIR` loads a local schema-v1 profile; with an HF model, `hf:default` or `hf:NAME` explicitly
  selects a bundled profile. Without the flag, bundled profiles are ignored. Requests select the loaded profile with
  `steering_profile` and `steering_scale`; `--steering-force SCALE` is the process-wide qualification mode.
- `--g6env-x "NAME=value ..."` passes engine settings to every rank ([docs/SETTINGS.md](docs/SETTINGS.md)). The
  boxes compare them, and a `G6_*` variable in the environment is refused for that reason.

### Remote KV

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

[engine/kvman-rs/REMOTE-TIER.md](engine/kvman-rs/REMOTE-TIER.md) covers the block stores, their transports and
measurements. [KNOBS.md](KNOBS.md) lists every `KV_STORE` and `KV_ROWS` field, and
[docs/SETTINGS.md](docs/SETTINGS.md) the engine's `G6_RT_*` settings.

### Thinking

GLM 5.3 always thinks: its chat template has no switch to turn thinking off. A chat request controls how much it thinks
with these fields:

| field | values | on GLM 5.3 |
|---|---|---|
| `reasoning_effort` (top level, or in `chat_template_kwargs`) | `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | The template names `low`, `high` and `max`. Another value becomes the nearest of those, ties going higher: `minimal` is `low`, `medium` is `high`, `xhigh` is `max`. Unset is `max`. `none` and any other value are a 400 |
| `thinking_budget` | an integer, or null | While the think block is open, the front end puts `\n</think>\n\n` in place of the reply's Nth token (so N-1 reasoning tokens come first), and the answer follows. 0, null or unset takes the server default, as in TensorFold. A negative number is no limit |
| `chat_template_kwargs.enable_thinking` | `true`; `false` with `--allow-thinking-off 1` | `false` is a 400 that names `reasoning_effort` `low`, `thinking_budget` and `--allow-thinking-off` instead |
| `response_format` | `{"type": "text"}`, `json_object`, `json_schema` (chat) | JSON modes are checked after generation, not constrained: the reply's first JSON value, and for `json_schema` the schema's XGrammar grammar ([frontend/README.md](frontend/README.md)) |

`usage.completion_tokens_details.reasoning_tokens` counts the reply's tokens through its `</think>`.
`--default-reasoning-effort` and `--default-thinking-budget` set the defaults for requests that name none (KNOBS.md).
The engine puts the close in place of the token itself (an insert rule, PROTOCOL.md), so a cut costs one more main step.
With an engine from before that, the front end ends the run and continues from the prompt, the reply and the close,
which prefills the prompt's last partial 4,096-token cell and the reasoning again: 2.5 to 4.2 s at TP=4's cold prefill
rate (frontend/README.md). A request that uses `logprobs` cannot have a budget.

## Troubleshooting

- "handshake refused": every box prints the field that differs and each box's value. It is an option, the model (a
  different commit or variant), the engine build, a box whose checks failed, or a box that does not answer on a rail.
- The boot stops with "GiB left after the weights and part of a KV pool": lower `--cap`, or use a smaller `--variant`.
- A box waits for the others: with mentat, check that every box's daemon sees the rest (`mentatd status`); with
  `--peers`, the workers wait for the head.
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
- Each rank checks its weights at boot: the GPU sums what it holds and compares with the bytes the loader read. A
  difference stops the boot with exit 3 and names the rank and the chunk ("g6: weight check: ..."). Once serving,
  `curl localhost:8100/health?deep` on the head checks every rank again ([frontend/README.md](frontend/README.md)).
  `G6ENV_X=G6_WEIGHT_CHECK=0` turns both off.
- `top_k` and `min_p` are ignored with a warning. `logprobs` and `prompt_logprobs` work up to k = 20.
  [frontend/README.md](frontend/README.md) has the full API.

| port | what | where |
|---|---|---|
| 8100 | the OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`) | head |
| 7340 | the supervisors: the handshake before boot, then status, logs and down | every box |
| 7341 | the multi-session control channel between the ranks | head |
| 18600-18659 | fabric bootstrap and the per-collective regions | head |

## Layout

| path | what |
|---|---|
| `gr8` | the entry point: `./gr8 glm53 <command>` |
| `tools/launch/` | the per-box supervisor behind `up`: options, the model and its cache, the handshake, mentat |
| `engine/runtime/` | `g6m4.cu` (one rank), `m4sim.cu` (one GPU simulating every rank), the step, prefill and the multi-session scheduler (`s2/`) |
| `engine/kernels/` | experts and dense trellis matrices, GEMV/GEMM, router, norms, sampling |
| `engine/attn/` | MLA decode and prefill attention and the sparse indexer |
| `engine/comms/` | the RoCE v2 verbs fabric: collectives, bulk proxy, context-parallel layout |
| `engine/loader-rs/` | the Rust loader: plans each rank's image from the release, fetches from Hugging Face, converts the drafter |
| `engine/kvman-rs/` | the Rust S2 page allocator: every rank's KV pages, live and retained runs, prefix-reuse matches |
| `frontend/` | the HTTP front end (`g6front-rs`) and the shim that drives the ranks |
| `scripts/` | build, tests and `engine_args.sh` (the engine command line) |
| `tools/convert/` | the readers, writers and test fixtures behind the exlr8 release |
| `smoketest/`, `bench/` | the smoke test, benchmarks and the quality gate |
| `docs/` | engine settings, the prefill attention design, the model-specific parts |
| [KNOBS.md](KNOBS.md), [KNOWN-ISSUES.md](KNOWN-ISSUES.md), [PROTOCOL.md](PROTOCOL.md), [AGENTS.md](AGENTS.md) | every option; open bugs; the front end to engine protocol; how to build, test and land a change |

gr8 is written for GLM 5.3's shapes, not as a general engine. [docs/MODEL-SPECIFIC.md](docs/MODEL-SPECIFIC.md) lists every
place the model is built in, and what would carry over to another model.

## Credits

KindlingAI is:

- [@adapt-ai-systems](https://github.com/adapt-ai-systems)
  - Prefill optimization, TP=6 S2 serving and context-budget improvements
    ([#7](https://github.com/kindlingai/gr8/pull/7), [#9](https://github.com/kindlingai/gr8/pull/9),
    [#27](https://github.com/kindlingai/gr8/pull/27), [#29](https://github.com/kindlingai/gr8/pull/29)).
  - Earlier engine, TP=3 conversion and gate foundations preserved in the Git history
    ([range fetching and manifests](https://github.com/kindlingai/gr8/commit/2d88926cb08bec602ba6382ccf10d90e1a735841),
    [gate harness](https://github.com/kindlingai/gr8/commit/3d6116f084b86aa96d4be47b253e223ef7a5dcea)).
- [@coffee-the-dev](https://github.com/coffee-the-dev)
- [@mmastrac](https://github.com/mmastrac)

The following contributions are greatly appreciated. If your name is missing, please file a PR: we want to ensure your work is recognized!

Thanks to:

- [@turboderp](https://github.com/turboderp)
  - EXL3 ideas, research and trellis quantization through [exllamav3](https://github.com/turboderp-org/exllamav3).
- [@OsakaTX](https://github.com/OsakaTX)
  - Launcher and container reliability, shell quoting and process handling
    ([#8](https://github.com/kindlingai/gr8/pull/8), [#16](https://github.com/kindlingai/gr8/pull/16),
    [#17](https://github.com/kindlingai/gr8/pull/17), [#21](https://github.com/kindlingai/gr8/pull/21)).
  - Setup documentation, input validation and HTTP defaults/connection limits
    ([#15](https://github.com/kindlingai/gr8/pull/15), [#18](https://github.com/kindlingai/gr8/pull/18),
    [#20](https://github.com/kindlingai/gr8/pull/20)).
- [@jakejharris](https://github.com/jakejharris)
  - Per-request activation steering, Ablit integration, profile tooling and qualification reports
    ([#45](https://github.com/kindlingai/gr8/pull/45) and its extracted steering PRs).
  - Ring relay backpressure and regression tests
    ([#58](https://github.com/kindlingai/gr8/pull/58), [#59](https://github.com/kindlingai/gr8/pull/59)).
- [@knapcio](https://github.com/knapcio)
  - Performance experiments and proposals for speculative decoding and INT4 draft weights
    ([#83](https://github.com/kindlingai/gr8/pull/83), [#85](https://github.com/kindlingai/gr8/pull/85),
    [#88](https://github.com/kindlingai/gr8/pull/88), [#89](https://github.com/kindlingai/gr8/pull/89)).
  - Fabric MTU negotiation, distributed GEMVs, MoE overlap, attention kernels and decode profiling proposals
    ([#82](https://github.com/kindlingai/gr8/pull/82), [#84](https://github.com/kindlingai/gr8/pull/84),
    [#86](https://github.com/kindlingai/gr8/pull/86), [#87](https://github.com/kindlingai/gr8/pull/87),
    [#90](https://github.com/kindlingai/gr8/pull/90)).
  - Shared-scope steering in S2 batched decode (the MoE shared-expert finalize of shared steps,
    [#209](https://github.com/kindlingai/gr8/pull/209)).
- [@eqvse](https://github.com/eqvse) and [@cl83](https://github.com/cl83)
  - TP=3/TP=4 ring fabric and relay support ([#4](https://github.com/kindlingai/gr8/pull/4)).
- [@chuck-ads](https://github.com/chuck-ads)
  - TP=4/TP=6 KL reference baselines and prompt-set benchmarking
    ([TP=4 baseline](https://github.com/kindlingai/gr8/commit/b671f958852bab1a139e9de5cd5904b368b3c5b1),
    [TP=6 baseline](https://github.com/kindlingai/gr8/commit/54c9161abaae5b317e74f26fc3bf9cf207535cf2),
    [prompt sets](https://github.com/kindlingai/gr8/commit/d9543184d9332996ab828bcffefcad892ed757e4)).
- [@spectra-the-bot](https://github.com/spectra-the-bot)
  - Host-memory reclamation and canonical session KV export/import
    ([#60](https://github.com/kindlingai/gr8/pull/60), [#65](https://github.com/kindlingai/gr8/pull/65)).
- [@unsaltedbutter-ai](https://github.com/unsaltedbutter-ai)
  - API compatibility and structured-output proposals ([#71](https://github.com/kindlingai/gr8/pull/71)).
- [@alexellis](https://github.com/alexellis)
  - [RigMark](https://github.com/alexellis/rigmark), used for decode and concurrency benchmarking.

### Upstream projects and research

- [exllamav3](https://github.com/turboderp-org/exllamav3) by [@turboderp](https://github.com/turboderp) and its contributors: the EXL3 format and its
  trellis quantizer, a streamlined variant of Cornell RelaxML's [QTIP](https://github.com/Cornell-RelaxML/qtip). This
  engine would not exist without their quantization work. The trellis decoders here were written from the EXL3 format,
  and the exlr8 weights were encoded with exllamav3's quantizer.
- [davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw), whose
  per-expert bit distribution `home` keeps expert for expert.
- [RigMark](https://github.com/alexellis/rigmark) by [@alexellis](https://github.com/alexellis), for decode and concurrency benchmarking.
- The tool-call cases in `frontend/g6front-rs/src/parser.rs` are ported from vLLM's GLM tool-parser tests (Apache-2.0).

Credits are also maintained in [CREDITS.md](CREDITS.md).

## License

MIT, see [LICENSE](LICENSE). The weights and steering profiles are not part of this repository and carry their own
licenses (GLM 5.3's license, each release's model card and notice, and the DFlash2 drafter's CC BY-NC-ND 4.0).
