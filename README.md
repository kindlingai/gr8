# GLM-5.3 (full) on DGX Spark (GB10): TP=4 or 6

Private preview. Feedback, issues and PRs are welcome; if you run it, tell us what breaks.

g6 is a from-scratch CUDA inference engine for the full GLM 5.3 model (MoE, 256 experts, MLA attention with the sparse
indexer, one MTP layer) on four or six NVIDIA DGX Spark (GB10) boxes, tensor-parallel 4 or 6 ways. It uses no vLLM,
PyTorch or NCCL at serve time: each box runs one `g6m4` process, the ranks talk over RoCE v2 through our own verbs
fabric, and an OpenAI-compatible front end runs on rank 0. What makes it fast:

- Several requests at once (`--s2`): up to 8 sessions share each decode step, with exact per-session output.
- DFlash2 speculative decoding for a request that runs alone, with the draft count chosen each round from a live model
  of step cost and acceptance. Sessions that share steps use the MTP layer.
- Pair-owner QO attention in prefill at TP=4 (`G6_PF_OWN=4`, [docs/DESIGN-QO-PAIR.md](docs/DESIGN-QO-PAIR.md)).
- exlr8 experts: the routed experts at 3.25 bpw, trellis-decoded inside the expert kernels, loaded straight from their
  width files.
- Context-parallel KV past 32,768 tokens, so a 393K (TP=4) or 614K (TP=6) pool fits.
- Prefix reuse across sessions (`REUSE=1`): a request that continues an earlier conversation re-prefills only the new
  part, with the same output as a cold run.

| | TP=4 (S2, DFlash2) | TP=6 (K1H2, one request at a time) |
|---|---|---|
| prefill, 8K prompt, cold | 1,293 tok/s | 1,942 tok/s |
| decode\*, code / prose / structured | 49.5 / 28.4 / 68.4 tok/s | prose 36.9 tok/s (MTP k=2, temp 0.7) |
| aggregate\*, 2 / 4 streams | 50.6 / 63.6 tok/s | – |
| TTFT, long prompts | 18K: 14.5 s, 87K: 73.3 s | 131K: 79.1 s, 254K: 176.1 s |
| KV pool | 393,216 tokens | 614,400 tokens |
| requests decoding at once | 8 | 1 |
| prefix reuse, a continued 20K conversation | 7.1 s vs 17.3 s cold | – |

\* TP=4: [RigMark](https://github.com/alexellis/rigmark) 1.0.0 (`reasoning_effort` low), GPU clocks capped at
2200 MHz. TP=6: one test window on one cluster; two arms with identical decode code differed by about 3% on prose, so
read small differences as noise.

TP=4 is the default: `scripts/launch_tp4.sh` boots the multi-session engine (`scripts/launch_s2.sh`). TP=6 serves one
request at a time through `scripts/launch_tp6.sh`. [docs/SETTINGS.md](docs/SETTINGS.md) explains every setting,
[CHANGELOG.md](CHANGELOG.md) has what changed and the measurements behind it, and
[KNOWN-ISSUES.md](KNOWN-ISSUES.md) has the open bugs.

## What you need

- **Four or six DGX Spark or other GB10 boxes** (sm_121a, 128 GB unified memory). One rank per box, and nothing else
  holding GPU memory: a TP=4 rank holds about 90 GB of weights.
- **A RoCE v2 fabric with two rails.** Each box's ConnectX-7 has two addresses (`IPS` and `IPS2` in the config), one
  per PCIe root. We used one 200G switch. The engine finds the RDMA device for each address itself.
- **Passwordless ssh from the head (rank 0) to the other boxes.** The launcher, the shim and the front end run on the
  head.
- **Disk per box:** the box's rank pack (about 90 GB at TP=4, 62 GB at TP=6), the drafter pack at TP=4 (0.86 GB) or
  the shared QO pack at TP=6 (13 GB), and the binary.
- **Disk for the pack build** (one machine, any Linux box or one Spark): about 340 GB for the TR3 download, 10 GB for
  the GPTQ codes and about 390 GB for the packs it writes.
- **Toolchain:** CUDA 13 (`nvcc` with `sm_121a`), g++ with C++20, rdma-core / libibverbs-dev (the stock DGX OS,
  Ubuntu 24.04), and cargo for the Rust pack loader and front end.
- **The weights**, public on Hugging Face:
  [davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw) (the routed
  experts, plus the tokenizer files the front end reads) and
  [adapt-ai-systems/glm-5.3-full-exl3-gptq-dense](https://huggingface.co/adapt-ai-systems/glm-5.3-full-exl3-gptq-dense)
  (the dense GPTQ codes, `tp4/` or `tp6/`).

## Ports

| port | what | where |
|---|---|---|
| 8100 | the OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`) | head (`PORT`) |
| 7341 | the multi-session control channel between the ranks | head (`S2PORT`) |
| 18600-18659 | fabric bootstrap and the per-collective regions | head |

## Layout

| path | what |
|---|---|
| `engine/runtime/` | `g6m4.cu` (one rank on the fabric), `m4sim.cu` (one GPU simulating all the ranks), the rank step, prefill, the serve loops, `build_m4.sh` |
| `engine/runtime/s2/` | the multi-session scheduler, pool, dispatch and control channel |
| `engine/kernels/` | experts (TR3 / exlr8 trellis), dense GEMV/GEMM (MXFP8, int4), glue (router, norms, sampling) |
| `engine/attn/` | MLA decode and prefill attention and the sparse indexer |
| `engine/comms/` | the RoCE v2 verbs fabric: collectives, bulk proxy, CP layout |
| `engine/loader/`, `engine/loader-rs/` | the pack reader, and the Rust loader that fills the device image |
| `tools/convert/` | public weights to per-rank packs (`convert.py`), the QO pack (`qopack.py`), the drafter pack (`dflash2.py`), the checker (`checkpack.py`) |
| `tools/tune/` | fetching the exlr8 v2 expert files |
| `frontend/` | the HTTP front end (`g6front-rs`, Rust, and `g6front`, Python) and the shims that drive the ranks (`g6s2shim.py`, `g6shim.py`) |
| `PROTOCOL.md` | the front end to engine socket protocol |
| `scripts/` | build, pack build, stage, launch, stop, tests |
| `bench/` | benchmarks and the quality gate |
| `docs/` | settings, the QO pair design, the model-specific parts |

## 1. Describe the cluster

    cp config/cluster.env.tp4.example config/cluster.env     # or cluster.env.example for TP=6
    $EDITOR config/cluster.env

It names the boxes (`HOSTS`, rank order), their fabric addresses (`IPS`, `IPS2`), the working directory on every box
(`G6_HOME`, the same path everywhere), the tokenizer directory on the head (`MODEL_DIR`) and the API port (`PORT`).
Every script reads it; `G6_CLUSTER=<file>` points them at another one.

## 2. Build the engine

On one Spark (a few minutes, CPU only):

    G6_NR=4 scripts/build.sh bin/        # TP=4, with the multi-session path
    scripts/build.sh bin6/               # TP=6
    scripts/build_front.sh bin/          # the Rust front end

`BUNDLE_LIBS=1` also copies cuBLAS next to the binary, for boxes without the CUDA toolkit. The `-D` flags in
`scripts/build.sh` pick kernel variants; each one won an A/B test on our cluster.

## 3. Build the packs

On the machine with the disk (CPU only, about 15-25 minutes):

    hf download davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw --local-dir /data/tr3
    hf download adapt-ai-systems/glm-5.3-full-exl3-gptq-dense --include 'tp4/*' --local-dir /data/gptq
    pip install numpy
    scripts/build_packs.sh /data/tr3 /data/gptq/tp4 /data/packs

At TP=4, also build the DFlash2 drafter packs, `packs/dflash.r<r>.*` on every box
(`python3 -m tools.convert.dflash2 --src <drafter dir> --out /data/packs --nr 4`). The launcher stops with instructions
when they are missing; `DFLASH=0` runs MTP only.

The exlr8 experts come from their own width files: fetch them with `tools/tune/fetch_ranks.py`, then boot with
`V2=<dir> V2_SLICES=<dir> V2_WIDTHS=<manifest>` (`tools/tune/README.md`).

## 4. Stage

    scripts/stage.sh bin/ /data/packs

This copies the binary and each box's packs onto that box. On the head, also create the front end's venv once:

    python3 -m venv $G6_HOME/venv && $G6_HOME/venv/bin/pip install -r frontend/requirements.txt

## 5. Start it

From the head (an `ssh -A` session works if the head has no key of its own for the others):

    scripts/launch_tp4.sh                # or scripts/launch_tp6.sh
    scripts/stop.sh

The first boot checks the sha256 of every pack tensor; later boots can skip that with `NOHASH=1`. Useful knobs, all
environment variables of the launcher: `SESS` (concurrent sessions, default 8), `REUSE=1` (prefix reuse), `DFLASH=0`
(MTP only), `SPEC_ACCEPT=rejection`, `CAP` (the KV pool). `PLAIN=1 scripts/launch_tp4.sh` boots the older
one-request loop, from a `PLAIN=1 G6_NR=4 scripts/build.sh` build; it is being retired.

Logs are in `$G6_HOME/out/` on each box (`g6m4.r<r>.err`), plus `shim.log` and `front.log` on the head.

## 6. Check it

    curl http://<head>:8100/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

    python3 bench/pbench.py --url http://<head>:8100 --tag mine

`pbench.py` reports 8K and 32K cold prefill and prose decode, and appends one JSON line to `bench/results.jsonl`.

## Tuning

Every engine setting is an environment variable or a launcher flag, each explained in
[docs/SETTINGS.md](docs/SETTINGS.md). The ones that matter most:

- **`CAP`** is the KV pool in tokens, and so the longest context. Lower it first if a box runs short of memory.
- **`SESS`** is how many requests decode at once (1-8). More sessions capture more step graphs (about 2.5 GiB at 8).
- **`G6_PF_OWN`** picks the prefill attention: 4 at TP=4 (pair-owner QO), 3 at TP=6 with the QO pack (row-owner QO).
  Prompts shorter than `G6_PF_QP_MIN` (768) take the cheaper OWN=2 path.
- **`G6_CP=1 G6_CP_R=32768`** is context-parallel KV: the first 32,768 tokens are replicated on every rank, later pages
  are spread across ranks. `G6_CP_PF_VERIFY=1` and `G6_CP_PF_HASH2=1` check every staged KV page against a hash.
- **`MTP`** is the MTP draft count for sessions that share steps; DFlash2's count is chosen per round.

## Troubleshooting

- **Decode slows past the replicated window.** Past `G6_CP_R` tokens the KV is split across ranks and every step
  gathers from all of them.
- **A request past 32,768 tokens waits for an empty pool.** It runs alone at the start of the pool, so it waits for
  every resident session to finish, and nothing else is admitted while it runs.
- **A request with `prompt_logprobs` holds the others.** Its whole prompt runs as decode steps (about 13 ms per prompt
  token at TP=4) before anyone else's next step; the shim caps such prompts at 1,024 tokens.
- **If a rank dies, the whole engine stops.** Restart with `scripts/stop.sh && scripts/launch_tp4.sh`.
- **Requests reach the other ranks as files over ssh** (one persistent connection per box), which adds a little TTFT.
- **Boot takes a few minutes with hashing on.** It is mostly pack read speed: about 1.8 GB/s per box for us.
- `top_k` and `min_p` are ignored with a warning; `logprobs` and `prompt_logprobs` work up to k = 20.
  `frontend/README.md` has the full list.
- Code comments refer to internal design notes and test-window names that are not in this repository.

## What has and has not been verified

| item | status |
|---|---|
| TP=4 launch | `launch_tp4.sh` / `launch_s2.sh` / `stop.sh` run against the real engine on four Sparks. |
| TP=6 launch | `launch_tp6.sh` / `stop.sh` / `stage.sh` run end to end with a fake engine (`frontend/tests/fake_serve_engine.py`) standing in for six ranks on one host. **Not yet run against the real engine on six Sparks**; our own TP=6 boots used an older harness with the same env and args. |
| `build_packs.sh` | The GPTQ input dirs on Hugging Face match the layer-000 hashes our converter expects. The full conversion was **not** re-run from these scripts, and we did not record which revision of the TR3 repo we downloaded. Run `checkpack.py` (the script does). |
| exactness | Multi-session output equals each request run alone, byte for byte (logprobs included); DFlash2 under `--s2` equals the one-request loop's; a prefix-reuse hit equals a cold run. A full KL / GSM8K / HumanEval gate is not reported here. |

## Other models

g6 is written for GLM 5.3's shapes, not as a general engine. [docs/MODEL-SPECIFIC.md](docs/MODEL-SPECIFIC.md) lists
every place the model is baked in (the converter's dims and rank map, the kernels' fixed sizes, attention, the layer
sequence, the front end's tokens and parser) and what carries over unchanged (fabric, loader, pack format, serve loop,
front end). Use it as the map for porting another model.

## Credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp and its contributors: the EXL3 format and its
  trellis quantizer, a streamlined variant of Cornell RelaxML's [QTIP](https://github.com/Cornell-RelaxML/qtip). This
  engine would not exist without their quantization work. The TR3 decoder in `engine/kernels/tr3*.cuh` was written
  from the EXL3 format, and the exlr8 expert weights were encoded with exllamav3's quantizer.
- The routed-expert weights are the [davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw)
  EXL3 quantization of GLM 5.3, including its per-expert bit distribution: in each layer the 64 experts with the
  highest K3 round-trip error are encoded at K4 and the other 192 at K3, 3.25 bpw on average. The exlr8 re-encode keeps
  that distribution expert for expert.
- [RigMark](https://github.com/alexellis/rigmark) by alexellis, for the decode and concurrency numbers above.
- The tool-call cases in `frontend/tests/test_parser.py` are ported from vLLM's GLM tool-parser tests (Apache-2.0).

## License

MIT, see [LICENSE](LICENSE). The weights are not part of this repository. They carry their own licenses (GLM 5.3's
license, and the EXL3 quantization's model card).
