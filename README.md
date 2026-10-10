# GLM-5.3 (full) on DGX Spark (GB10): TP=4 or 6

Private preview. Feedback, issues and PRs are welcome; if you run it, tell us what breaks.

g6 is a from-scratch CUDA inference engine for the full GLM 5.3 model (MoE, 256 experts, MLA attention with the sparse
indexer, one MTP layer) on four or six NVIDIA DGX Spark (GB10) boxes, tensor-parallel 4 or 6 ways. It uses no vLLM,
PyTorch or NCCL at serve time: each box runs one `g6m4` process, the ranks talk over RoCE v2 through our own verbs
fabric, and an OpenAI-compatible front end runs on rank 0. What makes it fast:

- Several requests at once (`--s2`): up to 8 sessions share each decode step, with exact per-session output.
- Speculative decoding chosen per round: a request that runs alone drafts with DFlash2 or the MTP layer, whichever is
  expected to commit more tokens per ms (DFlash2 wins on code, MTP on prose), with the draft count chosen from a live
  model of step cost and acceptance. Sessions that share steps use the MTP layer.
- Pair-owner QO attention in prefill at TP=4 (`G6_PF_OWN=4`, [docs/DESIGN-QO-PAIR.md](docs/DESIGN-QO-PAIR.md)).
- exlr8 experts: the routed experts at 3.25 bpw, trellis-decoded inside the expert kernels, loaded straight from their
  width files.
- Context-parallel KV past 32,768 tokens, so a 393K (TP=4) or 614K (TP=6) pool fits.
- Prefix reuse across sessions (on by default at TP=4, `--reuse 0` turns it off): a request that continues an earlier conversation re-prefills only the new
  part, with the same output as a cold run.

| | TP=4 (S2, DFlash2) | TP=6 (K1H2, one request at a time) |
|---|---|---|
| prefill, 8K prompt, cold | 1,278 tok/s | 1,942 tok/s |
| decode\*, code / prose | 49.4 / 30.1 tok/s | prose 36.9 tok/s (MTP k=2, temp 0.7) |
| aggregate\*, 2 / 4 streams | 50.6 / 62.7 tok/s | – |
| TTFT, long prompts | 18K: 14.5 s, 87K: 73.3 s | 131K: 79.1 s, 254K: 176.1 s |
| KV pool | 393,216 tokens | 614,400 tokens |
| requests decoding at once | 8 | 1 |
| prefix reuse, a continued 20K conversation | 7.1 s vs 17.3 s cold | – |

\* TP=4: [RigMark](https://github.com/alexellis/rigmark) 1.0.0 (`reasoning_effort` low), GPU clocks capped at 2200 MHz,
main at 4964737 booted with `./gr8 glm53 up`. RigMark's structured row is left out: it measures predictable output, not
agent speed. TP=6: one test window on one cluster; two arms with identical decode code differed by about 3% on prose, so
read small differences as noise.

TP=4 is the default and boots the multi-session engine. TP=6 serves one request at a time.
[KNOBS.md](KNOBS.md) lists every launch knob and [docs/SETTINGS.md](docs/SETTINGS.md) every engine setting,
[model.yaml](model.yaml) has the checkpoints and the memory footprint, [CHANGELOG.md](CHANGELOG.md) has what changed
and the measurements behind it, and [KNOWN-ISSUES.md](KNOWN-ISSUES.md) has the open bugs.
[kindling.json](kindling.json) has the table above in a form a program can read; a PR that changes the table changes
it too.

## Quick start

On every box, with the boxes, the fabric and the packs ready ([What you need](#what-you-need)):

    git clone https://github.com/kindlingai/gr8 && cd gr8
    ./gr8 glm53 build                  # the engine for this TP and the Rust front end, into bin/ (or copy bin/)
    ./gr8 glm53 packs /data/tr3 /data/gptq/tp4 /data/packs /data/dflash2    # once, on one machine (step 3)
    ./gr8 glm53 up --tp 4 --peers 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
        --model https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4 --model-cache /opt/g6/cache \
        --base-packs /data/packs --drafter https://huggingface.co/incoai/GLM-5.3-DFlash2
    ./gr8 glm53 status                 # every box, the front end, /health
    ./gr8 glm53 smoke                  # questions with answers that can be checked

With `--model https://huggingface.co/...`, each box fetches only its own part of the exlr8 release into
`--model-cache` (step 4): its expert halves for the variant, its index slices, the manifests, the dense and vocab
files, and the tokenizer. Until the engine reads the release's dense files (#116), it still takes its non-expert
weights from base packs (`--base-packs`, step 3). `--drafter` names the DFlash2 drafter's own repo. The drafter's licence
forbids sharing a converted copy, so each box fetches the drafter and converts its rank. Without `--drafter`, TP=4 runs
MTP only.

Every box runs the same `up`. The first address in `--peers` is the head. The head runs the API. The boxes compare their
options, model set and engine build before any engine starts, and all of them stop, naming the difference, if one box
disagrees. `--variant` picks the expert widths (`home`, the default, is 3.25 bpw). Without `--v2` the experts come from
the TR3 rank packs instead, at the same widths but without exlr8's decode path. The same options can sit in
`cluster.env` on each box (`config/cluster.env.tp4.example`), so `./gr8 glm53 up` alone is enough. Without `--peers`,
the boxes find each other through the local mentat daemon (`--mentat localhost`, the default).

For six boxes, use six peers and the GPTQ `tp6/` codes. Every launch knob is a flag, such as `--sess 4` or `--reuse 0`
([KNOBS.md](KNOBS.md)). `./gr8 glm53 down` on any box stops every box, `./gr8 glm53 restart` stops and boots again, and
`./gr8 glm53 logs [rank]` shows a rank's engine log from any box.

TP=3 has a `G6_NR=3` build and engine settings, but no measurements yet, and `scripts/build_packs.sh` builds TP=4 and TP=6
packs only (`tools/convert/convert.py --nr 3` for the rest).

## What you need

- Four or six DGX Spark or other GB10 boxes (sm_121a, 128 GB unified memory). One rank per box, and nothing else
  holding GPU memory: a TP=4 rank holds about 90 GB of weights.
- A RoCE v2 fabric with two rails. Each box's ConnectX-7 has two addresses (`IPS` and `IPS2` in the config), one
  per PCIe root. We used one 200G switch. The engine finds the RDMA device for each address itself.
- A LAN between the boxes, with TCP port 7340 open on it and on both fabric rails (the supervisors' handshake and
  control port).
- Non-interactive ssh from the head (rank 0) to every other box's LAN address, with each address already in the head's
  `known_hosts`. The shim writes each request into every rank's directory over ssh. That is the one ssh use left.
- Python 3 on every box (the supervisor uses the standard library only), and the venv for the shim on the head.
  The fetch from Hugging Face needs nothing more than `bin/g6load`, which `./gr8 glm53 build` builds with the engine
  (engine/loader-rs): it plans each rank's ranges from the index slices and checks every half's xxhash64. Only
  `--drafter` needs numpy, to convert the drafter.
- Disk per box: the box's rank pack (about 90 GB at TP=4, 62 GB at TP=6), the drafter pack at TP=4 (0.86 GB) or
  the shared QO pack at TP=6 (13 GB), and the binary.
- Disk for the pack build (one machine, any Linux box or one Spark): about 340 GB for the TR3 download, 10 GB for
  the GPTQ codes and about 390 GB for the packs it writes.
- Toolchain: CUDA 13 (`nvcc` with `sm_121a`), g++ with C++20, rdma-core / libibverbs-dev (the stock DGX OS,
  Ubuntu 24.04), and cargo for the Rust pack loader and front end.
- The weights, public on Hugging Face:
  [davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw) (the routed
  experts, plus the tokenizer files the front end reads) and
  [adapt-ai-systems/glm-5.3-full-exl3-gptq-dense](https://huggingface.co/adapt-ai-systems/glm-5.3-full-exl3-gptq-dense)
  (the dense GPTQ codes, `tp4/` or `tp6/`).

## Ports

| port | what | where |
|---|---|---|
| 8100 | the OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`) | head (`PORT`) |
| 7340 | the supervisors: the handshake before boot, then status, logs and down | every box, LAN (`CONTROL_PORT`) |
| 7341 | the multi-session control channel between the ranks | head (`S2PORT`) |
| 18600-18659 | fabric bootstrap and the per-collective regions | head |

## Layout

| path | what |
|---|---|
| `gr8` | the entry point (`./gr8 glm53 <command>`): `up`, `down`, `restart`, `status`, `logs`, `check`, `show-config`, `prepare`, `smoke`, `bench`, `build`, `stage`, `packs` |
| `tools/launch/` | the per-box supervisor behind `up`: options, the model set and its cache, the handshake, mentat |
| `config/` | `cluster.env` examples for TP=4 and TP=6 |
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
| `scripts/` | build, pack build, stage, tests, `engine_args.sh` (the engine command line), and the deprecated ssh launchers |
| `smoketest/` | `run.sh <base> [served-name]`, or `./gr8 glm53 smoke` |
| `bench/` | benchmarks and the quality gate |
| `docs/` | settings, the QO pair design, the model-specific parts |
| `model.yaml` | the checkpoints and the per-rank memory footprint |
| `KNOBS.md` | every launch, `./gr8 glm53` and build knob |
| `kindling.json` | the results table, machine-readable |
| `AGENTS.md` | how to build, test, launch and land a change |

## 1. Describe the cluster

Either on the command line of every box, or in a file on every box:

    cp config/cluster.env.tp4.example cluster.env     # or cluster.env.example for TP=6
    $EDITOR cluster.env

It names the boxes (`PEERS`, LAN addresses in rank order with the head first, or `MENTAT=localhost` and `TP`), the model
(`MODEL`, a pack directory, and `MODEL_CACHE`, this box's local disk), the working directory (`G6_HOME`, by default the
checkout) and the API port (`PORT`). `./gr8 glm53` reads `@FILE` from the command line, else `$G6_CLUSTER`, else
`./cluster.env`, else `config/cluster.env`. `./gr8 glm53` parses the file and does not source it. The command line wins over the file, the
file over the environment, the environment over the defaults. `./gr8 glm53 show-config` prints the result and where each
value came from. An old file still works: `HOSTS` is read as `PEERS` (then every box must resolve its entries: use
addresses or DNS names, and no ssh aliases), and `IPS`/`IPS2` pin the fabric addresses.

Each box finds its own fabric addresses: the two IPv4 addresses with a RoCE v2 GID on an active port, the lower one rail
0. Set `IPS` and `IPS2` (every rank's addresses) to pin them.

## 2. Build the engine

On one box (a few minutes, CPU only), then copy `bin/` to the same place on the others, or build on each box. Every
box must run the same `bin/g6m4`: the handshake compares its sha256.

    ./gr8 glm53 build

It runs `G6_NR=4 scripts/build.sh bin/` at TP=4 (with the multi-session path), `scripts/build.sh bin/` at TP=6 and
`G6_NR=3 scripts/build.sh bin/` at TP=3, then `scripts/build_front.sh bin/` (the Rust front end). On a box with no
cluster file or `--tp`, name the TP: `TP=4 ./gr8 glm53 build`. The scripts still work on their own.

`BUNDLE_LIBS=1` also copies cuBLAS next to the binary, for boxes without the CUDA toolkit. The `-D` flags in
`scripts/build.sh` pick kernel variants; each one won an A/B test on our cluster.

## 3. Build the packs

On the machine with the disk (CPU only, about 15-25 minutes):

    hf download davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw --local-dir /data/tr3
    hf download adapt-ai-systems/glm-5.3-full-exl3-gptq-dense --include 'tp4/*' --local-dir /data/gptq
    pip install numpy
    ./gr8 glm53 packs /data/tr3 /data/gptq/tp4 /data/packs /data/dflash2

`./gr8 glm53 packs` runs `scripts/build_packs.sh` (the rank count follows the GPTQ dir: `tp4/` or `tp6/`) and, at TP=4
with a drafter dir (a `config.json` and `model.safetensors`), the DFlash2 drafter packs, `dflash.r<r>.*` (`python3 -m
tools.convert.dflash2 --src <drafter dir> --out /data/packs --nr 4`), and copies the four tokenizer files into
`/data/packs/tokenizer/`. Without the drafter packs, boot with `--dflash 0` for MTP only. `./gr8 glm53 check` says which.

The pack directory is the model set. `--model` needs every rank's manifest, every rank's drafter manifest at TP=4
(`dflash.r<r>.manifest.json`) and the tokenizer files in it, even though each box loads only its own rank. A per-box
`packs/` from `./gr8 glm53 stage` holds one rank's files only, so it cannot serve as `--model`: point every box at the
whole pack directory, on a shared mount or a copy. `set.json` describes it (every rank's manifest hash and pack size, the drafter and
QO packs, the tokenizer files) and its sha256 is the set id the boxes compare. The boxes need not see it at the same
path. Downloading from Hugging Face and preparing from a raw checkpoint inside `up` are not in this version.

The exlr8 experts come from their own width files: fetch them with `tools/tune/fetch_ranks.py`, then boot with
`--v2 <dir> --v2-slices <dir>` (`tools/tune/README.md`). The cache holds the base packs only. The width files stay
where they are. `--variant` picks the width manifest:

| variant | routed experts |
|---|---|
| `home` (the default) | 3.25 bpw, the TR3 checkpoint's widths |
| `down1` | about 2.25 bpw |
| `up1` | about 4.0 bpw |

A name means `exlr8-manifest-<name>.json` in the model set, or in the `--v2` directory (directly or under
`manifests/`). A path is used as it is (`--v2-widths` is the older spelling). The variant's name and manifest hash go
into the id the boxes compare, so two boxes with different variants refuse to boot together, while the cache keeps one
copy of the base packs for all variants. Without `--v2`, only `home` is accepted. `up` and `show-config` print the
expert GB on the busiest rank under the variant's widths. Whether that fits next to the KV pool is up to you: lower
`--cap` for `up1`.

## 4. The model cache

From Hugging Face. `--model https://huggingface.co/<org>/<repo>` (optionally `@<rev>` or `/tree/<rev>`; anything that
does not start with `https://` is a local path) fetches into the cache:

    DIR/hf/<org>--<repo>/refs/<rev>             the commit the revision resolved to
    DIR/hf/<org>--<repo>/<commit>/manifests/, tokenizer/, experts/, index-slices/tp<N>/k<K>/rank<r>.safetensors,
        dense/, complete/tp<N>.rank<r>.<variant>.json

`experts/` holds this rank's expert halves for the variant at their file offsets (sparse files, about 75 GB at TP=4
home), fetched with range GETs (`tools/tune/fetch_ranks.py`), with the ranges planned from this rank's index slices
and every half checked against its xxhash64 by `bin/g6load`. Every whole file
is checked against the release's `hashes/` before it is renamed into place. The dense and vocab files (about 11 GB) are
fetched whole for now. A fetch resumes where it stopped, and the `complete/` marker is written last. `up` uses the
commit recorded at the last resolve, so a boot works offline once the cache holds its part. `prepare` asks the hub
again. The commit goes into the id the boxes compare, so boxes on different commits refuse to boot together. `HF_TOKEN`
is sent only when it is set. The published repos are public.

`--drafter https://huggingface.co/<org>/<repo>` (TP=4) fetches the DFlash2 drafter into `DIR/hf/<org>--<repo>/<commit>/`
and converts this box's rank there with `tools/convert/dflash2.py`. The drafter we use is
`https://huggingface.co/incoai/GLM-5.3-DFlash2` (CC BY-NC-ND 4.0, so no converted copy is shared). With DFlash2 on and no
drafter, `up` runs MTP only and prints the flag to add, and `--dflash 1` without one is refused.

    ./gr8 glm53 prepare --tp 4 --peers 10.0.0.1,... --model https://huggingface.co/... --model-cache /opt/g6/cache
    ./gr8 glm53 prepare --tp 4 --rank 2 --model https://huggingface.co/... --model-cache /opt/g6/cache   # mentat

fills the cache ahead of `up`, with a progress line every 30 s (GB done of the total, the rate). With `--peers` the box
knows its rank. With mentat it does not until the election, so name it with `--rank`.

From a pack directory.

`--model-cache DIR` is a directory on this box's own disk. At boot, once a box knows its rank, it copies only its own
files from `--model` into it: its rank pack and manifest, its drafter pack at TP=4, the QO pack where `PF_OWN=3`, and the
tokenizer files:

    DIR/sets/<set id>/set.json, tokenizer/, rank<r>/{rank<r>.*, dflash.r<r>.*, qo.*, COMPLETE}

A copy goes through `DIR/tmp/` and a rename, and `COMPLETE` is written last, so an interrupted copy is never used. Every
boot checks `COMPLETE`, each file's size and the manifests' sha256 against `set.json`. The engine checks every tensor's
sha256 at load unless `NOHASH=1`. A later boot with `--model-cache DIR` alone (no `--model`) uses the one set the cache
holds for this TP. Copying a 90 GB rank pack from a shared mount can take an hour. Run `./gr8 glm53 prepare --model ... --model-cache ...`
on each box before the first `up`, so the boot finds the cache filled. Without `--model-cache`, the engine loads from `--model` in place.

`./gr8 glm53 stage` (ssh, from the head) still copies `bin/` and each box's packs into `$G6_HOME`, for the `--ssh`
launchers, and creates the front end's venv on the head when it is missing (`python3 -m venv $G6_HOME/venv &&
$G6_HOME/venv/bin/pip install -r frontend/requirements.txt`).

## 5. Start it

On every box:

    ./gr8 glm53 up [OPTIONS]
    ./gr8 glm53 down

`up` starts this box's supervisor in the background (`$G6_HOME/out/supervisor.log`) and follows it until the cluster
serves. The supervisor:

1. reads the options and the model set, and finds the boxes. A model set offers the ranks whose files it holds: one
   rank for a box that holds one rank's packs, any rank for a whole pack directory or a cache with `--model`. With
   `--peers`, this box's rank is the index of its own address, and a box listed at a rank it does not hold fails its
   checks, with the position it belongs at. With `--mentat`, each box registers in the mentat group `gr8-glm53` and
   publishes its offer. Once the group has `--tp` live members on two reads 5 s apart, the boxes that offer particular
   ranks take them, and the rest take the free ranks in address order (`--head` pins rank 0). When no assignment fits,
   every box stops and prints each box's offer.
2. runs this box's checks (the binary, `/dev/infiniband`, an engine still running, the front end and the venv on the
   head) and copies its shard into the cache.
3. handshakes with the head on port 7340. Each box sends its options, model set id, engine build and failed checks, then
   asks for the verdict about once a second. The head compares each hello with its own, checks that it reaches every box
   on both rails, and answers with the boot epoch and every rank's fabric addresses. On any difference every box prints
   the field and each box's value, and exits 4. A worker waits for a slow head as long as the head answers. The head
   refuses the boot when a worker stops asking for 30 s, and it answers only the members' addresses.
4. starts its own `g6m4` with the command line from `scripts/engine_args.sh`. The head also starts the front end, waits
   for the engines, starts the shim, and in mentat mode announces the API to mentat (agent `gr8-glm53@gr8-api@<head>`).
   The engine gets its own variables plus `PATH`, `HOME`, `USER`, `LANG`, `LD_LIBRARY_PATH`, `CUDA_*` and `NVIDIA_*`
   from the environment. A `G6_*` variable in the environment stops the boot: pass it with `--g6env-x` so every box
   compares it.
5. stays: `status`, `logs` and `down` reach every box through the supervisors. When a rank's engine exits, its
   supervisor tells the head, and the head stops every box.

`--shape ring` is refused: the engine has no ring fabric yet. In mentat mode the supervisor also publishes its state in
mentat (`gr8.state`, `gr8.rank`, `gr8.set`, `gr8.opts`), so `mentatd status` shows it.

The first boot checks the sha256 of every pack tensor. Later boots can skip that with `--nohash 1`. Useful knobs:
`--sess` (concurrent sessions, default 8), `--reuse 0` (prefix reuse off, it is on by default), `--dflash 0` (MTP only), `--cap` (the KV pool).
[KNOBS.md](KNOBS.md) has the rest.

Logs are in `$G6_HOME/out/` on each box (`supervisor.log`, `g6m4.r<r>.err`), plus `shim.log` and `front.log` on the
head: `./gr8 glm53 logs [rank|shim|front|supervisor|all] [-f]`, from any box.

The ssh launchers still work for one release: `./gr8 glm53 --ssh up` runs the old checks and `scripts/launch_tp4.sh`
(which runs `scripts/launch_s2.sh`), `launch_tp6.sh` or `launch_tp3.sh` from the head, and `--ssh down`, `--ssh status`
and `--ssh logs` work as before. The ssh launchers read `HOSTS`, `IPS`, `IPS2` and `MODEL_DIR`, and the config file wins over the
environment there. They build the engine command line from the same `scripts/engine_args.sh`.

## 6. Check it

    ./gr8 glm53 status
    ./gr8 glm53 smoke
    ./gr8 glm53 bench --tag mine

`status` asks every box's supervisor (its state, engine, MemAvailable, variant, the last log line of a stopped engine), and shows
the head's shim, front end and `/health`. `smoke` runs `smoketest/run.sh` against the head's API: cases with answers that can be
checked (a fact, arithmetic with thinking on and off, streaming, a tool call, logprobs, the context limit), because a
wrong pack can load cleanly and serve fluent nonsense. Its exit status is the number of failures. `bench` runs
`bench/pbench.py`, which reports 8K and 32K cold prefill and prose decode, and appends one JSON line to
`bench/results.jsonl`. By hand:

    curl http://<head>:8100/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

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

- Decode slows past the replicated window. Past `G6_CP_R` tokens the KV is split across ranks and every step
  gathers from all of them.
- A request past 32,768 tokens waits for an empty pool. It runs alone at the start of the pool, so it waits for
  every resident session to finish, and nothing else is admitted while it runs.
- A request with `prompt_logprobs` holds the others. Its whole prompt runs as decode steps (about 13 ms per prompt
  token at TP=4) before anyone else's next step; the shim caps such prompts at 1,024 tokens.
- If a rank dies, the whole engine stops. `./gr8 glm53 status` names it and shows its last log line; `./gr8 glm53
  restart` boots again.
- The boot stops with "handshake refused". Every box prints the field that differs and each box's value: an
  option, the model set id (the packs, drafter or tokenizer differ), the engine build (`bin/g6m4` differs), a box whose
  checks failed, or a box that does not answer on a fabric rail.
- A box waits for the others. With mentat it says "waiting for 4 members in group gr8-glm53, have 3" every minute:
  check that every box's mentat daemon sees the others (`mentatd status`). With peers the workers wait for the head.
- Requests reach the other ranks as files over ssh (one persistent connection per box), which adds a little TTFT.
- Boot takes a few minutes with hashing on. It is mostly pack read speed: about 1.8 GB/s per box for us.
- `top_k` and `min_p` are ignored with a warning; `logprobs` and `prompt_logprobs` work up to k = 20.
  `frontend/README.md` has the full list.
- Code comments refer to internal design notes and test-window names that are not in this repository.

## What has and has not been verified

| item | status |
|---|---|
| TP=4 launch | `launch_tp4.sh` / `launch_s2.sh` / `stop.sh` run against the real engine on four Sparks. |
| `./gr8 glm53` (symmetric) | Four supervisors on one Linux machine (127.0.0.1-4, with stub engines that play the S2 file protocol, the real shim, a stub ssh and a stub front end): boot, a request through the shim to every rank's directory, `status`, `logs`, `down` from a worker, an option mismatch, a missing shard, a rank that dies, and mentat mode against fake daemons. The agent link was also run against a real mentatd. On four Sparks: peers and mentat mode both boot and serve (31-43 s from `up` to the API with a warm page cache), smoke passes, `down` and `logs` work from a worker, and an option mismatch stops every box. This is how our cluster runs. |
| `./gr8 glm53 --ssh` | `check`, `status`, `logs`, `smoke`, `build`, `stage` and `packs` exercised against stub ssh, rsync and a fake API on one machine. The launchers' engine command lines are unchanged by the move into `scripts/engine_args.sh` (compared word for word under a stub ssh, for each TP and a range of knobs). |
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
- The routed-expert weights are the
  [davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw) EXL3 quantization
  of GLM 5.3, including its per-expert bit distribution: in each layer the 64 experts with the highest K3 round-trip
  error are encoded at K4 and the other 192 at K3, 3.25 bpw on average. The exlr8 re-encode keeps that distribution
  expert for expert.
- [RigMark](https://github.com/alexellis/rigmark) by alexellis, for the decode and concurrency numbers above.
- The tool-call cases in `frontend/tests/test_parser.py` are ported from vLLM's GLM tool-parser tests (Apache-2.0).

## License

MIT, see [LICENSE](LICENSE). The weights are not part of this repository. They carry their own licenses (GLM 5.3's
license, and the EXL3 quantization's model card).
