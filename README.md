# GLM-5.3 (full) on DGX Spark (GB10)

Early release. If you run it, tell us what breaks: issues and PRs are welcome.

g6 is a from-scratch CUDA inference engine for the full GLM 5.3 model (MoE with 256 experts, MLA attention with the
sparse indexer, one MTP layer) on three, four or six DGX Spark (GB10) boxes, tensor-parallel across all of them. It
uses no vLLM, PyTorch or NCCL at serve time. Each box runs one `g6m4` process, the ranks talk over RoCE v2 through
their own verbs fabric, and an OpenAI-compatible API runs on the first box.

## Results

| | TP=4 | TP=3 | TP=6 |
|---|---|---|---|
| experts | `home`, 3.25 bpw | `down1`, 2.25 bpw | `home`, 3.25 bpw |
| drafting | DFlash2 or MTP | MTP | MTP |
| decode\*, code / prose | 49.3 / 30.0 tok/s | 36.9 / 28.4 tok/s | prose 36.9 tok/s† |
| aggregate\*, 2 / 4 streams | 50.8 / 64.0 tok/s | 46.1 / 58.3 tok/s | – |
| prefill, 8K prompt, cold | 1,374 tok/s | 648 tok/s | 1,942 tok/s† |
| prefill, 8K prompt, cached | 54,867 tok/s | 53,747 tok/s | – |
| KV pool (longest context) | 393,216 tokens | 307,200 tokens | 614,400 tokens |
| requests decoding at once | 8 | 8 | 1† |

\* [RigMark](https://github.com/alexellis/rigmark) 1.0.0 (`reasoning_effort` low), GPU clocks capped at 2200 MHz,
booted from the Hugging Face release with `./gr8 glm53 up`. RigMark's structured row is left out: it measures
predictable output, not agent speed. TP=4 code varies between boots (41.5 to 49.3 on one build), because which requests
get DFlash2 depends on timing.
† TP=6 still serves one request at a time; it moves to the multi-session path next. Its numbers are from that loop,
MTP k=2 at temperature 0.7, one test window on one cluster.

[kindling.json](kindling.json) has this table in a form a program can read. A PR that changes one changes the other.

## How it works

- One model: the [exlr8 release](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) on Hugging Face. Each box
  fetches only its own part, once, and boots from its cache in about 40 s. Each rank plans its weights from the
  release's files at load. The routed experts and the dense trellis matrices are decoded inside the kernels.
- Several requests at once (TP=4 and TP=3): up to 8 sessions share each decode step. Each request's output is byte for byte what it
  would be alone, logprobs included.
- Speculative decoding: every round drafts with the MTP layer, or at TP=4, for a request that runs alone, with the
  DFlash2 drafter when that is expected to commit more tokens per ms (DFlash2 wins on code, MTP on prose). The draft
  count comes from a live model of step cost and acceptance.
- Prefix reuse: a request that continues an earlier conversation re-prefills only the new part, with the same output
  as a cold run. A continued 20K conversation takes 7.1 s instead of 17.3 s at TP=4.
- Prefill attention split by query and head owner across ranks ([docs/DESIGN-QO-PAIR.md](docs/DESIGN-QO-PAIR.md)).
- Context-parallel KV: the first 32,768 tokens are on every rank, later pages are spread across ranks, so the pools
  above fit.

## What you need

- Three, four or six DGX Spark or other GB10 boxes (sm_121a, 128 GB unified memory), one rank per box, with nothing
  else holding much memory. A rank's weights take about 83 GB at TP=4 and 78 GB at TP=3. The KV pool and the step
  graphs need most of the rest.
- A RoCE v2 fabric with two rails: each box's ConnectX-7 has one address on each PCIe root. We use one 200G switch.
  At TP=3 or 4, `--shape ring` runs without one: each box's f0 port is cabled to the next box's f1 port, and at TP=4
  the diagonal's writes are relayed by the box between, through a 32 MB ring per fabric (about 0.25 GB pinned per
  rank). It costs about 10% of single-stream decode (4-5% at 4 streams). memlock must be unlimited.
- A LAN between the boxes with TCP ports 7340 and 7341 open. TP=4 and TP=3 need no ssh between the boxes: the head
  engine passes each request to the other ranks over its control links. TP=6's one-request loop still writes each
  request to the other ranks over non-interactive ssh from the first box (their addresses in its `known_hosts`).
- Disk per box: about 90 GB at TP=4 (its experts, the 11 GB of dense and vocab files, the drafter) and about 85 GB at
  TP=3. One cache holds both, because their parts share the sparse expert files.
- Proactive memory compaction off on every box. A rank fills the box to about 2 GB under its limit, and at that level
  the kernel's background compaction moves pages under the engine in bursts that delay every collective. As root,
  once (it holds across reboots; `vm.compaction_proactiveness=20` undoes it):

      echo 'vm.compaction_proactiveness = 0' > /etc/sysctl.d/90-g6-no-proactive-compaction.conf
      sysctl -w vm.compaction_proactiveness=0

- To build: CUDA 13 (`nvcc` with `sm_121a`), g++ with C++20, rdma-core / libibverbs-dev, cargo, and Python 3. The
  stock DGX OS (Ubuntu 24.04) has all of them except cargo.

## Quick start

On every box:

    git clone https://github.com/kindlingai/gr8 && cd gr8
    TP=4 BUNDLE_LIBS=1 ./gr8 glm53 build
    ./gr8 glm53 up --tp 4 --peers 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
        --model https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4 --model-cache /opt/g6/cache \
        --drafter https://huggingface.co/incoai/GLM-5.3-DFlash2

Then, from any box:

    ./gr8 glm53 status        # every box, the front end, /health
    ./gr8 glm53 smoke         # questions with answers that can be checked
    curl http://10.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

For three or six boxes, build with `TP=3` or `TP=6`, pass `--tp 3` or `--tp 6` with as many peers, and leave out
`--drafter`: the DFlash2 drafter's 8 KV heads do not split three or six ways. `BUNDLE_LIBS=1` copies cuBLAS next to the
engine, for boxes without the CUDA toolkit. Every box must run the same build: build on one and copy `bin/`, or build
on each.

The first `up` downloads this box's part: about 35 minutes at 40 MB/s. To download ahead of time, run `./gr8 glm53
prepare` with the same options. The drafter's licence (CC BY-NC-ND 4.0) forbids sharing a converted copy, so each box
downloads the drafter and converts its own rank.

## Running it

Every box runs the same `./gr8 glm53 up`. The first address in `--peers` is the head: it runs the API on port 8100.
Before any engine starts, the boxes compare their options, model and engine build, and if one box differs, every box
stops and prints the difference. Without `--peers`, the boxes find each other through a local
mentat daemon (`--mentat localhost`, the default).

| command | what it does |
|---|---|
| `up [OPTIONS]` | start this box's supervisor and follow it until the cluster serves |
| `down`, `restart` | stop every box, or stop and boot again |
| `status` | every box's state, engine and free memory, and the head's front end and `/health` |
| `logs [rank\|shim\|front\|supervisor] [-f]` | a log from any box |
| `prepare [OPTIONS]` | fill `--model-cache` with this box's part, without booting |
| `check`, `show-config` | what would stop this box's boot; every option and where it came from |
| `smoke`, `bench` | the smoke test, and prefill and decode benchmarks, against the head's API |
| `build` | the engine and the Rust front end, into `bin/` |

Options come from the command line, then a config file (`@FILE`, else `./cluster.env`; see
`config/cluster.env.tp4.example`, `.tp3.example`, or `cluster.env.example` for TP=6), then the environment. With the
options in `cluster.env`, `./gr8 glm53 up` alone is enough. [KNOBS.md](KNOBS.md) lists every option.

When a rank's engine exits, the head stops every box: `status` names the rank and shows its last log line. Logs are in
`out/` on each box (`supervisor.log`, `g6m4.r<r>.err`, and `shim.log` and `front.log` on the head).

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
| `down1` | about 2.25 bpw | TP=3 |
| `up1` | about 4.0 bpw (lower `--cap`) | |

`home` does not fit at TP=3: it puts 108.7 GB of weights on each rank. A boot that would run a box out of memory stops
while it allocates the KV pool, and names `--cap` and `--variant`.

### Options worth knowing

- `--cap` is the KV pool in tokens, and so the longest context. Lower it first when a box runs short of memory.
- `--sess` is how many requests decode at once (1-8). Each one adds step graphs: about 2.5 GiB at 8.
- `--reuse 0` turns prefix reuse off.
- `--dflash 0` runs MTP only at TP=4. `--mtp` is the MTP draft count (2).
- `--g6env-x "NAME=value ..."` passes engine settings to every rank ([docs/SETTINGS.md](docs/SETTINGS.md)). The
  boxes compare them, and a `G6_*` variable in the environment is refused for that reason.

## Troubleshooting

- "handshake refused": every box prints the field that differs and each box's value. It is an option, the model (a
  different commit or variant), the engine build, a box whose checks failed, or a box that does not answer on a rail.
- The boot stops with "GiB left after the weights and part of a KV pool": lower `--cap`, or use a smaller `--variant`.
- A box waits for the others: with mentat, check that every box's daemon sees the rest (`mentatd status`); with
  `--peers`, the workers wait for the head.
- Decode slows past 32,768 tokens of context: from there the KV is spread across ranks, and every step gathers from
  all of them.
- A request past 32,768 tokens waits for the pool to empty, and runs alone.
- A request with `prompt_logprobs` runs its prompt as decode steps (about 13 ms a token at TP=4) and holds the others
  meanwhile. The shim caps those prompts at 1,024 tokens.
- `top_k` and `min_p` are ignored with a warning. `logprobs` and `prompt_logprobs` work up to k = 20.
  [frontend/README.md](frontend/README.md) has the full API.

| port | what | where |
|---|---|---|
| 8100 | the OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`) | head |
| 7340 | the supervisors: the handshake before boot, then status, logs and down | every box |
| 7341 | the multi-session control channel between the ranks | head |
| 18600-18659 | fabric bootstrap and the per-collective regions | head |

## Packs (optional)

Before the release, the model was converted into per-rank packs from the TR3 expert checkpoint
([davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw)) and GPTQ dense
codes ([adapt-ai-systems/glm-5.3-full-exl3-gptq-dense](https://huggingface.co/adapt-ai-systems/glm-5.3-full-exl3-gptq-dense)).
Packs still boot: `./gr8 glm53 packs TR3_DIR GPTQ_DIR OUT_DIR [DRAFTER_DIR]` builds them (about 390 GB written, CPU
only), and `--model OUT_DIR` boots from them. Most users should use the release.

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
| `frontend/` | the HTTP front end (`g6front-rs`) and the shim that drives the ranks |
| `scripts/` | build, tests and `engine_args.sh` (the engine command line) |
| `tools/convert/` | packs from public checkpoints (optional) |
| `smoketest/`, `bench/` | the smoke test, benchmarks and the quality gate |
| `docs/` | engine settings, the prefill attention design, the model-specific parts |
| [KNOBS.md](KNOBS.md), [CHANGELOG.md](CHANGELOG.md), [KNOWN-ISSUES.md](KNOWN-ISSUES.md), [PROTOCOL.md](PROTOCOL.md), [AGENTS.md](AGENTS.md) | every option; what changed and the measurements behind it; open bugs; the front end to engine protocol; how to build, test and land a change |

g6 is written for GLM 5.3's shapes, not as a general engine. [docs/MODEL-SPECIFIC.md](docs/MODEL-SPECIFIC.md) lists every
place the model is built in, and what would carry over to another model.

## Credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp and its contributors: the EXL3 format and its
  trellis quantizer, a streamlined variant of Cornell RelaxML's [QTIP](https://github.com/Cornell-RelaxML/qtip). This
  engine would not exist without their quantization work. The trellis decoders here were written from the EXL3 format,
  and the exlr8 weights were encoded with exllamav3's quantizer.
- [davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw), whose
  per-expert bit distribution `home` keeps expert for expert.
- [RigMark](https://github.com/alexellis/rigmark) by alexellis, for the decode and concurrency numbers above.
- The tool-call cases in `frontend/tests/test_parser.py` are ported from vLLM's GLM tool-parser tests (Apache-2.0).

## License

MIT, see [LICENSE](LICENSE). The weights are not part of this repository and carry their own licenses (GLM 5.3's
license, the release's model card, and the DFlash2 drafter's CC BY-NC-ND 4.0).
