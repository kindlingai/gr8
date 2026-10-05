> ## Welcome
>
> **Custom GLM 5.3 engine for 6× DGX Spark. Private preview.**
>
> - **What it is:** an inference engine for GLM 5.3 full written from scratch. No vLLM; our own CUDA kernels plus a small server. It runs across 6 Sparks.
> - **Where it's at (6 Sparks, one session):**
>   - ~1,950 tok/s prefill at 8K (TTFT ~4.2 s)
>   - ~37 tok/s prose decode
>   - 614K context; a 254K prompt prefills in ~176 s
> - **What's in the repo:**
>   - engine source
>   - a converter that builds the weight packs from public HF weights
>   - one-command launch/stop scripts
>   - docs on every setting
>   - a map of what's GLM-specific vs reusable (for other models)
> - **Known limits:**
>   - one request at a time (multi-session is in progress)
>   - decode slows past the replicated context window (being worked on)
>   - launch scripts not yet run outside our own setup
>   - the 4-Spark (TP4) version runs a split form of the QO prefill path (`G6_PF_OWN=4`, see Known limits)
> - **Where help would be great:**
>   - multi-session at long context
>   - decode cost with context split across nodes
>   - trying it on other models
> - Feedback, issues and PRs all welcome. If you run it, tell us what breaks.

See [CHANGELOG.md](CHANGELOG.md) for what changed since the K1H2 snapshot and [KNOWN-ISSUES.md](KNOWN-ISSUES.md) for open bugs.

# g6: a custom GLM 5.3 (full) engine for six DGX Sparks

g6 is a from-scratch CUDA inference engine for the full GLM 5.3 model (MoE, 256 experts, MLA attention with the sparse
indexer, one MTP layer) on six NVIDIA DGX Spark (GB10) nodes, tensor-parallel 6 ways (TP6). It does not use vLLM,
PyTorch or NCCL at serve time. Each node runs one `g6m4` process. The ranks talk to each other over RoCE v2 through our
own verbs fabric. An OpenAI-compatible HTTP front end runs on rank 0.

The weights are the community EXL3 "TR3" 3.25 bpw quantization for the routed experts. Dense decode uses GPTQ int4 g32
codes and prefill uses MXFP8 copies of the original BF16 weights. These are the same weights an EXL3/vLLM build serves,
re-packed into one file per rank.

This snapshot is the **K1H2** build: our current best six-node configuration, with context-parallel KV past 32K
tokens and a 614,400-token KV pool.

**Status: research code, private preview.** It runs on our cluster. The scripts here are new; see
[What has and has not been verified](#what-has-and-has-not-been-verified) before you plan around it.

## Measured numbers (our cluster, K1H2, one boot, single request)

| metric | K1H2 |
|---|---|
| prefill, 8K prompt (`bench/pbench.py`) | 1,942 tok/s |
| prose decode, single stream, MTP k=2, temp 0.7 | 36.9 tok/s |
| TTFT at 131,072 prompt tokens | 79.1 s |
| TTFT at 253,952 prompt tokens | 176.1 s |
| KV pool (`--cap`) | 614,400 tokens |
| lowest MemAvailable on any node, while serving | 13.7 GB |

All numbers come from one test window on one cluster. In that window, two arms with identical decode code differed by
about 3% on prose decode, so read small differences as noise. These are not controlled comparisons with other stacks.

## Hardware

- **6 × DGX Spark** (GB10, 128 GB unified memory). One rank per node, and nothing else holding GPU memory.
- **RoCE v2 fabric with two rails.** Each node's ConnectX-7 has two addresses (`IPS` and `IPS2` in the config), one per
  port/HCA. We used one 200G switch at MTU 1500. The engine finds the RDMA device for each address itself.
- **Disk per node:** about 62 GB for the node's rank pack, 13 GB for the shared QO pack, and the binary.
- **Disk for the pack build (one machine, any Linux box or one Spark):** about 340 GB for the TR3 download, 10 GB for
  the GPTQ codes, and about 390 GB for the packs it writes.
- **The head (rank 0)** needs passwordless ssh to the other five nodes. The front end and shim run there.
- **Toolchain we built with:** CUDA 13.0.88 (`nvcc` with `sm_121a`), g++ 13.3, rdma-core / libibverbs-dev 50 (the stock
  DGX OS, Ubuntu 24.04).

## Quick start

```bash
# 0. config: node names + fabric addresses + working dir (same path on every node)
cp config/cluster.env.example config/cluster.env && $EDITOR config/cluster.env

# 1. build the engine on one Spark (CUDA 13 with sm_121a, libibverbs-dev; a few minutes, CPU only)
scripts/build.sh bin/

# 2. download the two public inputs and build the packs (CPU + disk, no GPU; ~15-25 min for us)
huggingface-cli download davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw --local-dir /data/tr3
huggingface-cli download adapt-ai-systems/glm-5.3-full-exl3-gptq-dense --include 'tp6/*' --local-dir /data/gptq
pip install numpy
scripts/build_packs.sh /data/tr3 /data/gptq/tp6 /data/packs

# 3. copy the binary and each node's packs onto that node
scripts/stage.sh bin/ /data/packs

# 4. front end venv on the head (once)
python3 -m venv $G6_HOME/venv && $G6_HOME/venv/bin/pip install -r frontend/requirements.txt

# 5. ONE command: boot all six ranks, then the shim and the HTTP server
scripts/launch_tp6.sh
curl http://<head>:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

scripts/stop.sh
```

The first boot checks the sha256 of every pack tensor at load. Later boots can skip that with `NOHASH=1 scripts/launch_tp6.sh`.

Benchmark: `python3 bench/pbench.py --url http://<head>:8100 --tag mine`. It reports 8K/32K cold prefill and prose
decode, and appends one JSON line to `bench/results.jsonl`.

## The settings that matter

Every engine setting is an environment variable or a flag in `scripts/launch_tp6.sh`. Each one is explained in
[docs/SETTINGS.md](docs/SETTINGS.md). In short:

- **`--cap`** (`CAP=`) is the KV pool in tokens, and so the longest context. 614,400 is what K1H2 was tested at. Lower it
  first if a node runs short of memory.
- **`--mtp`** (`MTP=`) is the MTP draft length. K1H2 was measured at k=2. Larger k is supported but was not measured on
  this build.
- **`G6_PF_OWN=3` + the QO pack** is row-owner attention in prefill. It is the biggest prefill lever and needs `qo.pack` on
  every node. TP4 runs `G6_PF_OWN=4` instead, the same attention block split over two head pairs, which needs no QO
  pack ([docs/DESIGN-QO-PAIR.md](docs/DESIGN-QO-PAIR.md)).
- **`G6_CP=1 G6_CP_R=32768`** is context-parallel KV. The first 32,768 tokens are replicated on every rank and later pages
  are spread across ranks, which is what makes the 614K pool fit. `G6_CP_PF_VERIFY=1` and `G6_CP_PF_HASH2=1` check every
  staged KV page against a hash. HASH2 makes that check cheap (254K TTFT -34% vs the old hash, same output tokens).
- **The compile flags in `scripts/build.sh`** pick kernel variants. Each one won an A/B test on our cluster.

## Repository map

| path | what |
|---|---|
| `engine/runtime/` | `g6m4.cu` (one rank on the fabric), `m4sim.cu` (one GPU simulating all six ranks), the rank step, prefill, serve loop, `build_m4.sh` |
| `engine/kernels/` | expert (TR3 trellis), dense GEMV/GEMM (MXFP8, int4), glue (router, norms, sampling) |
| `engine/attn/` | MLA decode + prefill attention and the sparse indexer |
| `engine/comms/` | RoCE v2 verbs fabric: collectives, bulk proxy, CP layout |
| `engine/loader/` | pack reader and the device image loader |
| `engine/ref/g6ref.py` | NumPy reference of the record layouts (the converter uses it) |
| `tools/convert/` | public weights -> per-rank packs (`convert.py`), QO pack (`qopack.py`), checker (`checkpack.py`) |
| `frontend/` | OpenAI/vLLM-compatible HTTP server (`g6front`) + the shim that drives the six ranks (`g6shim.py`) |
| `PROTOCOL.md` | the front end <-> engine socket protocol |
| `scripts/` | build, pack build, stage, launch, stop |
| `docs/` | settings, model-specific parts (porting to other models) |

## Other models

g6 is written for GLM 5.3's shapes, not as a general engine. [docs/MODEL-SPECIFIC.md](docs/MODEL-SPECIFIC.md) lists every
place the model is baked in: the converter's dims and rank map, the kernels' fixed sizes, attention (MLA + indexer),
the layer sequence, and the front end's tokens and parser. It also says what carries over unchanged (fabric, loader,
pack format, serve loop, front end shape). Use it as the map for porting another model.

## What has and has not been verified

| item | status |
|---|---|
| engine source = what we serve | The tree is our K1H2 source with only comments edited. `scripts/build.sh`'s flags built on a GB10 (exit 0, no errors) and gave SASS identical to our served binary, except two compiler-generated anonymous-namespace names. |
| `launch_tp6.sh` / `stop.sh` / `stage.sh` | Run end to end with a fake engine (`frontend/tests/fake_serve_engine.py`) standing in for six ranks on one host: all ranks ready, a chat request streamed, stop left no processes. **Not yet run against the real engine on six Sparks.** Our own boots use an older harness with the same env and args. `launch_tp4.sh` / `stop.sh` were run against the real engine on four Sparks (#2). |
| `build_packs.sh` | The six GPTQ input dirs on Hugging Face match the layer-000 hashes our converter expects. The full conversion was **not** re-run from these scripts. Our packs came from the same `convert.py` + `qopack.py` with the same inputs, but we did not record which revision of the TR3 repo we downloaded. Run `checkpack.py` (the script does) and, if you can, compare outputs against ours. |
| quality | Each boot checks that all six ranks produce identical tokens. K1H2's output was token-identical to the same build with the old page hash, at 131K and 254K prompts. A full KL / GSM8K / HumanEval gate on K1H2 is not reported here. |

## Known limits

- **One request at a time.** The shim queues the rest. There is no batching across requests.
- Six ranks (TP6) or four (TP4). TP4: `G6_NR=4 scripts/build.sh`, `scripts/build_packs.sh` with the TP4 GPTQ codes
  (`tp4/r0..r3`), `scripts/stage.sh`, `scripts/launch_tp4.sh`. A TP4 rank pack is about 90 GB, so the 13 GB QO pack
  does not fit next to it. TP4 runs pair-owner QO attention instead (`G6_PF_OWN=4`, about 1.3-1.4 GB of device memory plus 84 MB pinned per rank) with a
  393,216-token pool. TP4 numbers (2200 MHz cap, launcher defaults): 8K prefill 1,318 tok/s, 32K 1,288 tok/s, 18K TTFT
  14.5 s, 87K TTFT 73.3 s, decode code/prose/structured 34.8/31.5/37.0 tok/s, 87K and 250K needles found (#39).
  With `G6_PF_OWN=1` (no QO attention): 8K 816 tok/s, 32K 654 tok/s.
- DFlash2 drafter, opt-in at TP4: `DFLASH=1 scripts/launch_tp4.sh` replaces MTP with the DFlash2 block drafter
  (`engine/runtime/dflash.cuh`, packs from `tools/convert/dflash2.py`). It is lossless and leaves more memory free than
  MTP, because the MTP layer is not built. With GLM 5.3's template the gate workloads are reasoning text, and there it
  does not beat MTP k=2. The drafter wins on JSON answers. MTP stays the default. Numbers in the PR.
- At temperature > 0, drafts are accepted when they equal the target's own sample (shared noise), so a seed gives the
  same text with or without speculation. `SPEC_ACCEPT=rejection` switches to rejection sampling with greedy drafts: the
  same output distribution, other text for a seed.
- `logprobs` and `prompt_logprobs` work up to k = 20 (ranks beyond the top 20 are null). `top_k`/`min_p` are ignored with a warning. The full list is in `frontend/README.md`.
- Requests are passed to the ranks other than 0 as files over ssh (one persistent connection per node), which adds a little TTFT.
- With hashing on, boot takes a few minutes. It is mostly pack read speed: local packs loaded at about 1.8 GB/s per node
  for us.
- If a rank dies, the whole engine stops. Restart with `scripts/stop.sh && scripts/launch_tp6.sh`.
- Code comments refer to internal design notes and test-window names that are not in this repository.

## Credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp and its contributors: the EXL3 format and its
  trellis quantizer, a streamlined variant of Cornell RelaxML's [QTIP](https://github.com/Cornell-RelaxML/qtip). This
  engine would not exist without their quantization work. The TR3 decoder in `engine/kernels/tr3*.cuh` was written
  from the EXL3 format, `exl3rate/exl3rate.cu` follows exllamav3's decode and numerics (MIT, see
  `exl3rate/LICENSE-exllamav3`), and the exlr8 expert weights were encoded with exllamav3's quantizer.
- The routed-expert weights are the [davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw)
  EXL3 quantization of GLM 5.3, including its per-expert bit distribution: in each layer the 64 experts with the
  highest K3 round-trip error are encoded at K4 and the other 192 at K3, 3.25 bpw on average. The exlr8 re-encode keeps
  that distribution expert for expert.
- The tool-call cases in `frontend/tests/test_parser.py` are ported from vLLM's GLM tool-parser tests (Apache-2.0).

## License

MIT, see [LICENSE](LICENSE). The weights are not part of this repository. They carry their own licenses (GLM 5.3's
license, and the EXL3 quantization's model card).
