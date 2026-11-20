# First boot: the questions to ask

Most people will have an agent (Claude Code, Codex or another) install and start gr8 for them the first time. If you
are that agent, read this file before you change a box or run the first `up`. First check whether the boxes already run
[Kindling OS](https://github.com/kindlingai/kindling-spark-os) and, if not, ask the person whether they want to install
it and tell them what is involved (below). Then ask the setup questions one at a time, with the numbers given for each.
Write the answers into `.env` next to `compose.yaml` for Docker Compose (`docker/.env` in a checkout;
[docker/.env.example](../docker/.env.example)) or `cluster.env` on bare metal
([config/cluster.env.example](../config/cluster.env.example)), show the person the file, and only then boot.
When the person has no preference, keep the default. Every default here is the exact model, except DeepSeek-V4.1-Flash's
prefill: `./gr8 ds41` runs the faster v2 prefill unless `EXACT=1` (Q2a).

## Before you start: Kindling OS

gr8 runs on stock DGX OS and on [Kindling OS (`kindling-spark-os`)](https://github.com/kindlingai/kindling-spark-os),
a minimal read-only OS image for GB10 boxes (DGX Spark, ASUS Ascent GX10) installed beside DGX OS and booted from GRUB.

If you can already reach the boxes, check `/etc/spark-os-version` (or `/run/sparkos/host`). When every box already runs
Kindling OS, tell the person and go on to [1. The setup](#1-the-setup-every-model). Otherwise, **before** configuring
gr8, ask whether they want to install Kindling OS and explain both what they get and what is involved:

> "Before we set up gr8, do you want to install Kindling OS (`kindling-spark-os`) on your boxes, or stay on stock DGX
> OS? Kindling OS is optional — gr8 runs on stock DGX OS as-is — and installs beside DGX OS without wiping it."

Tell the person **what they get**:

- **More free memory than DGX OS (up to ~4 GB across the system):** about 2 GiB in host/unified RAM from a
  Canonical-signed kernel with 64 KiB pages (`nvidia-64k`, the default flavour, with THP off), which gr8 sees right
  away as extra `MemAvailable` headroom, plus `dispramd` (`/run/dispram/dispram.sock`), which exposes the GPU's unused
  2 GiB display carveout to clients configured to borrow from it (such as vLLM's `dispram_vllm` plugin; gr8's default
  allocation path uses standard CUDA memory today).
- **A clean, read-only boot beside DGX OS:** every boot starts from the same ~1.6 GB EROFS image on the DGX OS disk
  under a RAM overlay, avoiding long-uptime OS drift. DGX OS stays installed as the rescue system and keeps managing
  GRUB. Login users (`uid >= 1000`), passwords, SSH keys, `sudoers.d`, host keys and network settings carry across, and
  `/home`, `/srv`, `/var/log`, `/var/lib/docker`, `/var/lib/containerd`, `/var/lib/spark-watchdog` and `/var/lib/nfs`
  are bind-mounted from the DGX OS disk.
- **Cluster services ready at boot:** Docker and the NVIDIA container toolkit, RDMA and ConnectX tools, NVIDIA's DGX
  tuning packages, `mentatd` (so gr8 boxes can discover each other and their `rdma`-tagged ConnectX ports with `TP`
  instead of listing `PEERS`), `lldpd`, automatic ConnectX RoCE MTU raising (to RoCE 4096 / link MTU 4200 when a port
  is at 1500), `spark-watchdog`, `spark-agent` (status page and MCP tools on `:8090`), and a live status console on
  `tty1`.

Tell the person **what is involved**:

- **Prerequisites on each box:** an ARM64 NVIDIA GB10 box booted by UEFI on DGX OS 7 (Ubuntu 24.04) with GRUB, `sudo`,
  Docker, internet access, and at least 10 GiB free on the DGX OS root disk (about 6 GB during build plus 1.6 GB per
  image; **48 GiB or more** free lets `install` also create the 16 GiB `/swap-64k.img` swap file for the 64 KiB kernel
  while leaving 32 GiB free).
- **Access and remote-login checks before rebooting:**
  - `setup.sh` asks to confirm physical fallback access (HDMI output, USB keyboard and power cycle) in case an early
    boot hangs.
  - At least one DGX OS user needs `~/.ssh/authorized_keys`, because the trial boot must be confirmed over SSH.
  - **Tailscale:** a box reached only over Tailscale must include Tailscale in the image, or the trial boot cannot be
    reached and reverts after 10 minutes:
    `--sources /etc/apt/sources.list.d/tailscale.list --packages tailscale --mounts /var/lib/tailscale`.
  - **Persistent paths:** anything outside `/home`, `/srv`, `/var/log`, `/var/lib/docker`, `/var/lib/containerd`,
    `/var/lib/spark-watchdog` and `/var/lib/nfs` sits in the RAM overlay unless bound with `--mounts DIR` (for example
    `--mounts /var/tmp` or `--mounts /opt`). Docker volumes under `/var/lib/docker` and paths under `/srv` or `/home`
    persist by default.
  - **Switch/link MTU:** if a ConnectX link is up at MTU 1500, Kindling OS raises it to 4200 for RoCE 4096 unless
    `--connectx-mtu 0` is passed; the switch or peer on that cable must accept 4200-byte frames.
- **Steps, time and rollback (about 5–10 minutes and one reboot per box):**
  1. Generate one cluster secret once (`openssl rand -hex 32`) and use that same key on every box in the cluster.
  2. On each box, clone `https://github.com/kindlingai/kindling-spark-os` (`--branch stable`) and run the read-only
     preflight first: `kindling-spark-os/setup.sh --check --secret <CLUSTER_KEY>` (plus Tailscale or `--mounts` flags
     if needed).
  3. Run `kindling-spark-os/setup.sh --secret <CLUSTER_KEY>`. It builds the setup container (a few minutes the first
     time) and installs the image (about 4 minutes), logs every change and its `undo:` command to
     `~/kindling-spark-os-setup.log`, and queues a one-shot GRUB **trial** boot without rebooting on its own.
  4. Reboot (`sudo systemctl reboot`). The trial boot automatically falls back to DGX OS on kernel panic or if nobody
     confirms it within **10 minutes**.
  5. Reconnect over SSH within 10 minutes, check the box, and run `sparkos-promote` to make Kindling OS the default
     boot. At any time later, `sparkos-rollback` switches the default boot back to DGX OS and reboots (`--no-reboot` to
     switch without rebooting).

Wait for the person's choice before touching any box:

- **If yes:** run `setup.sh --check` on every box first, report every warning (SSH keys, Tailscale, root disk space for
  `/swap-64k.img`, ConnectX MTU, extra `--mounts`), and get the person's explicit confirmation of physical fallback
  access and of the exact `setup.sh` command before running it. Over non-interactive SSH without a TTY, pass `--yes`
  only after the person has confirmed both prompts (without a TTY or `--yes`, `setup.sh` reads EOF, prints
  `Nothing changed.` and exits `0` without installing), and check the output for `Done: kindling spark-os` rather than
  trusting exit code `0` alone. Confirm with the person before each `sudo systemctl reboot`, and verify
  `sparkos-promote` succeeded on every box before moving on to [1. The setup](#1-the-setup-every-model). Even on
  Kindling OS, check `sysctl vm.compaction_proactiveness` on every box and set
  `sudo sysctl -w vm.compaction_proactiveness=0` if it is non-zero ([DEPLOY.md](DEPLOY.md#before-you-start)).
- **If no:** leave DGX OS untouched and continue straight to [1. The setup](#1-the-setup-every-model). Kindling OS can
  still be installed later without reinstalling gr8 or re-downloading weights stored in Docker volumes, `/srv` or
  `/home`.

## 1. The setup (every model)

These have no default ([DEPLOY.md](DEPLOY.md), "The two settings"):

1. **Which model?** `glm53` (GLM-5.3), `glm53f` (GLM-5.3-Flash) or `ds41` (DeepSeek-V4.1-Flash-1017). `./gr8 models`
   lists each model's release and the TP values it takes. `ds41`'s release is
   `https://huggingface.co/kindlingai/deepseek-v4.1-flash-exlr8-native`, with `TP=4`. For `ds41`, remove the example's
   `DRAFTER=` line (GLM's DFlash2 drafter): `ds41` drafts with its own MTP layers, and the check refuses a drafter.
2. **Which boxes?** Their addresses in the same order on every box (`PEERS`; the first one is the head and serves the
   API on port 8100), or mentat with `TP`.
3. **Where do the weights come from?** A Hugging Face release into `MODEL_CACHE`, a local copy of the release, or
   model-loader ([DEPLOY.md](DEPLOY.md#with-model-loader)).

For `ds41`, also set Engram on every box: `GR8_DS_ENGRAM=1` and `GR8_DS_ENGRAM_PREFIX` in `EXTRA_ENV`
([ENV.md](ENV.md)). It is part of the model, not a trade-off. Its prefix needs a token map that is in neither the repo
nor the release, so write it before the first `up`: [DEEPSEEK-BOOT.md](DEEPSEEK-BOOT.md) is the whole ds41 boot in
order.

Optional, every model: faster restarts. Put `GR8_GEN_CACHE=<dir>` in `EXTRA_ENV` on every box, a directory on a local
disk ([ENV.md](ENV.md)). The loader then keeps the tensors it computes at load and reads them back on the next boot
instead of computing them again. The first boot of an engine build writes them; a rebuild is a miss and writes them
again, and the loader keeps the two newest builds' entries. The served bytes are the same. Measured: GLM-5.3 at TP=6
(on an earlier build, before this port), 17.4 GB a rank, "engines ready" 93-96 s
without it, 97 s on the boot that writes it, 20-21 s after.
DeepSeek-V4.1-Flash at TP=4, 79.6 GB a rank: 84 s without it, 110 s on the boot that writes it, 16 s after.
GLM-5.3 at TP=4 (ring, 2026-11-06, an image built from the change that added the cache, #382): 4.8 GB a rank,
"engines ready" 71 s without it (the image before that change), 71 s on the boot that writes it, 68 s after. At TP=4 the computed tensors are a small part of the boot.

## 2. DeepSeek-V4.1-Flash (`ds41`): how many chats, and how long

Ask these two first: they decide how the memory is split, and the later answers depend on them. Start with what the
person will use it for, then give the numbers for the questions that matter to them.

> "What will you mostly run on it: one person or one agent at a time, several agents or people at the same time, or
> very long inputs (whole books, whole code bases)?"

- One at a time: keep the defaults of Q1 and Q2, and go on to section 3 (speed against exactness).
- Several at the same time: Q1.
- Very long inputs: Q2.

Measured on four GB10 boxes (TP=4, a switch), the exlr8 release, Engram on, Q3's switches on, unless a line says
otherwise. "Free memory" is `MemAvailable` a box; a range is the lowest and highest of the four boxes.

### Q1. How many chats at once?

> "How many chats should it run at the same time: 5, 10 or 16? The total speed stays about the same: more slots mean
> no chat waits for a free one, but each chat writes slower. A chat that arrives when every slot is busy waits."

- Default: 5 (`SESS=5` in `cluster.env`).
- 10: `SESS=10`, and `GR8_DS_SESS_DRAFTS=1` in `EXTRA_ENV`. 16: `SESS=16`, and the same word.
  `GR8_DS_SESS_DRAFTS=1` lets every chat keep drafting when more than 5 write at once; it drafts fewer tokens a
  chat as more chats write (5 at 1-5 chats, 2 at 9-10, 1 at 11-16; [ENV.md](ENV.md)).
- `SESS` takes any count from 1 to 32; 5, 10 and 16 above are the measured setups.
- Measured on `main` at b10aae4 (2026-11-03) with Q3's switches on, one boot each, two runs each (the range; the first
  of the two runs is a little slower). Short code prompts. "Each" is the median chat's writing speed once it has a
  slot; "together" is all chats; "first word" is the median wait for the first word, slot wait included.

  | chat slots | 1 chat, tokens/s | 8 chats: each / together | 16 chats: each / together | 16 chats: first word | free memory at ready |
  |---:|---:|---:|---:|---:|---:|
  | 5 | 162-164 | 68-69 / 249-254 (3 wait) | 60-65 / 241-258 (11 wait) | 5.4-5.5 s | 24.3-24.8 GiB |
  | 10 | 157-163 | 51-56 / 245-260 | 35-38 / 251-275 (6 wait) | 4.2-4.4 s | 22.5-22.8 GiB |
  | 16 | 155-163 | 51-54 / 243-260 | 34-36 / 251-272 | 4.2-4.4 s | 20.4-20.8 GiB |

  - The total stays at about 235-275 tokens/s from 8 chats up, whatever the slot count. More slots do not make the
    boxes write more; they share the same speed among more chats, so no chat sits idle waiting for a slot.
  - With 1 to 4 chats, the second runs differ by 0-2% between slot counts; the first runs were 3-8% slower with 10
    and 16 slots than with 5.
  - Answers: with one chat at a time, 10 and 16 slots give the same 75 answers as 5 slots (71 of 75 right); with
    every slot busy, 73 of 75 are right (10 at once and 16 at once).
  - Memory: each extra 262,144-token slot costs about 0.35 GiB a box at ready (the 5-slot reading is an earlier
    boot of the same build). The prefix cache then grows to `GR8_DS_CACHE_MIB` (4,096 MiB a box by default) as
    chats run.
  - Suggest 5 for one person or one agent at a time, 10 or 16 when several agents must all make progress at once.

### Q2. How long can one chat be?

> "How long may one chat get: 262,144 tokens (the default), or the model's full 1,048,576? A full 1M-token prompt
> takes about 7 minutes before the first word, and several very long chats at once use most of the memory."

- Default: 262,144 (`CAP=262144`). A request then holds at most 262,134 tokens (prompt and reply).
- Long: `CAP=1048576` (a request holds at most 1,048,566 tokens). The numbers below were measured with
  `GR8_DS_CACHE_MIB=2048` in `EXTRA_ENV`; add it to match them.
- Measured with 16 slots x 1,048,576 and the prefix cache at 2,048 MiB a box (`GR8_DS_CACHE_MIB=2048`):
  - Every slot gets its full length at boot: 9.8-10.0 GiB free a box at ready.
  - A 1,031,542-token prompt reads in about 417 s. Two or three such prompts at once share the reading: three at once
    all finish after about 1,250 s, as three one after another would.
  - Free memory: 3.2-3.4 GiB a box after one full 1M chat; at least 2.69 GiB a box while three ran at once. More than
    three at once was not tested.
  - Recall at 1M was unreliable in these tests: 2 of 7 different 1M prompts found all 3 facts hidden in them (one at
    10%, 50% and 90% of the prompt). Other hosted DeepSeek-V4.1-Flash servers also missed facts at 1M. Up to 262,144
    tokens, every run that completed found all 3.
  - When a snapshot in the prefix cache matches the start of a follow-up's prompt, the follow-up reads only the tokens
    after that cached part; otherwise it reads the whole prompt again. A 1M snapshot is 882 MiB a box: about 2 fit in
    2,048 MiB, about 4 in the 4,096 MiB default (`GR8_DS_CACHE_MIB` unset). A chat's first turn can keep 2 snapshots
    (prompt end and reply end), so this is not a fixed number of cached chats: at 2,048 MiB, only the last 1M chat
    stayed cached.

## 3. DeepSeek-V4.1-Flash (`ds41`): speed against exactness

Measured on four GB10 boxes (TP=4, a switch), the exlr8 release, Engram on. "Speed" is the decode rate of one chat in
tokens/s. "Quality" is a fixed 75-item check set (code, math, JSON, reasoning), the median of 3 runs. Each answer adds
words to `EXTRA_ENV` (`"A=1 B=2"`, every rank).

### Q2a. Exact output, or the faster prefill?

> "Do you want the exact model's output? The default reads prompts faster with the v2 prefill, but two of its parts
> (the shared expert and the v2 BF16 projections on the tensor cores) change the output bits, so the same prompt can get
> a different reply."

- Default: the v2 prefill (the release stack: 12,288-row chunks, one prefill lane, `GR8_DS_PF_V2_BF16_TC=7`).
  `GR8_DS_PF_V2_BF16_TC=0` in `EXTRA_ENV` rolls back the v2 BF16 tensor-core projections; `GR8_DS_PF_V2_W16K=0` in
  `EXTRA_ENV` steps back to 8,192 rows; `EXACT=1` keeps the two-lane 4,096-row exact path.
- Exact output: `EXACT=1` in `cluster.env`. Prompts then run on the old prefill path (two lanes, 4,096-row chunks).
- A 1M-token context with `GR8_DS_MHC_ROW_SHARD=1` in `EXTRA_ENV` also runs the old prefill path.

### Q3. Faster writing for a small quality loss?

> "Do you want DeepSeek to write faster: about 40-50% on long replies, 20% (code) to 50% (prose) on short ones? The
> cost: its output is no longer the exact model's, and our 75-item check went from 73 to 71 right."

- Default: no.
- Yes: `GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1 GR8_DS_MOE_TOPK_DECODE=4`.
- Measured against `main` at ca24fdd0 (2026-11-01; two boots), and again on `main` at b10aae4 (2026-11-03; one boot each
  way). Long replies:
  code +41-42%, prose +50-52%, structured output +23-25%. Short 256-token replies: code +21-23%, prose +51%,
  structured +14% (b10aae4). Prompt reading (prefill) stays inside the run-to-run spread (b10aae4: one benchmark
  +1-2%, another 3% lower). Quality 73 -> 71 of 75 both times.
- Half-way, one switch alone (one boot each, 72 of 75 for both):
  - `GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1`: long replies code +28%, prose +29%; short replies prose +20%, code
    within noise.
  - `GR8_DS_MOE_TOPK_DECODE=4`: short replies code +15%, prose +13%, structured +13%; long replies prose +17%,
    structured +16%, code within noise.

### Q4. A smaller output head?

> "Do you want about 2-4% faster writing? Some answers change (13 of 75 in our check); quality did not drop."

- Default: no.
- Yes: `GR8_DS41_HEAD_MX=1` (the decode head in MXFP8, a little over half the bytes).
- Measured: a decode round 3.7% shorter (4 boots), new-text decode +2.4%. Quality 71 -> 72 of 75; 13 of 75 answers
  changed.

### Q5. Several chats at once: first word sooner, or each chat faster?

> "When several chats start together, do you want each to show its first word sooner (about 3x with 4 chats, 5x with
> 8), if with 4 chats each one then writes about 9% slower?"

- Default: yes (the release stack sets both). With `EXACT=1`: no.
- No (each chat faster): `GR8_S2_ADMIT_HOLD=0 GR8_S2_TAIL_FIRST=0`. Yes with `EXACT=1`: `GR8_S2_ADMIT_HOLD=1 GR8_S2_TAIL_FIRST=1`.
- Measured with 4 chats: median first word 648 -> 209 ms; each chat's median rate 75.7 -> 68.6 tokens/s; all 4 chats
  together +15%. With 8 chats: first word 1,656 -> 310 ms; all 8 together +5.5%. The answers did not change.
  (Measured with Q3's switches on.)

### Q6. A long prompt arrives while other chats write: keep them smooth, or answer it sooner?

> "When someone sends a very long prompt while other chats are writing, should those chats keep writing smoothly (the
> default), or should the long prompt get its first word sooner?"

- Default: smooth. The launcher sets `GR8_S2_DECODE_PF` to the smaller of 1024 and the prefill width.
- Sooner: `GR8_S2_DECODE_PF=0`. Smoother still: `GR8_S2_DECODE_PF=512`.
- Measured with 4 chats writing and a new 34K-token prompt (Q3 and Q4 switches on):

  | `GR8_S2_DECODE_PF` | each chat while the prompt reads | longest pause | the long prompt's first word |
  |---|---:|---:|---:|
  | 1024 (default) | 17.9 tokens/s | 0.49 s | 13.8 s |
  | 0 | 10.7 tokens/s | 1.15 s | 10.2 s |
  | 512 | 23.6 tokens/s | 0.36 s | 18.6 s |

## 4. Tell the person (no question)

- A `ds41` request holds at most 262,134 tokens (prompt and reply) at the default `CAP` (Q2). A 34K-token prompt
  shows its first word in about 9 s, a 139K-token prompt in about 37 s; writing then runs at the same speed as after a
  short prompt.
- Speed depends on what is asked: prose decodes slower than code.

## 5. After the answers

1. Write `.env` next to `compose.yaml` (`docker/.env` with `-f docker/compose.yaml` in a checkout) for Docker Compose,
   or `cluster.env` on bare metal, and show it to the person.
2. Run `./gr8 <model> check` and `./gr8 <model> show-config` on each box (`docker compose run --rm gr8 check` and
   `show-config` in Compose): what would stop the boot, and where each value came from.
3. Boot every box (`docker compose up -d` in Compose, or `./gr8 <model> up` on bare metal; [DEPLOY.md](DEPLOY.md)),
   then run `smoke` on the head (`docker compose exec gr8 /opt/gr8/gr8 <model> smoke` or `./gr8 <model> smoke`). Each
   must pass.
4. `./gr8 <model> bench` on the head (or inside the container without SSH clock sampling:
   `docker compose exec gr8 /opt/gr8/gr8 <model> bench --hosts "" --json /var/lib/gr8/home/out/bench.json`) gives the
   speed of this setup ([bench/README.md](../bench/README.md)).

GLM-5.3 and GLM-5.3-Flash have no measured trade-off questions here yet.
