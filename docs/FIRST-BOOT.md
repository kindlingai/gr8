# First boot: the questions to ask

Most people will have an agent (Claude Code, Codex or another) start gr8 for them the first time. If you are that
agent, read this file before the first `up`. Ask the person the questions below, one at a time, with the numbers given
for each. Write the answers into `cluster.env` ([config/cluster.env.example](../config/cluster.env.example)), show the
person the file, and only then boot. When the person has no preference, keep the default: every default here is the
exact model.

## 1. The setup (every model)

These have no default ([DEPLOY.md](DEPLOY.md), "The two settings"):

1. **Which model?** `glm53` (GLM-5.3), `glm53f` (GLM-5.3-Flash) or `ds41` (DeepSeek-V4.1-Flash-1017). `./gr8 models`
   lists each GLM model's release and the TP values it takes. It does not list `ds41`: its release is
   `https://huggingface.co/kindlingai/deepseek-v4.1-flash-exlr8-native`, with `TP=4`. For `ds41`, remove the example's
   `DRAFTER=` line (GLM's DFlash2 drafter): `ds41` drafts with its own MTP layers, and the check refuses a drafter.
2. **Which boxes?** Their addresses in the same order on every box (`PEERS`; the first one is the head and serves the
   API on port 8100), or mentat with `TP`.
3. **Where do the weights come from?** A Hugging Face release into `MODEL_CACHE`, a local copy of the release, or
   model-loader ([DEPLOY.md](DEPLOY.md#with-model-loader)).

For `ds41`, also set Engram on every box: `GR8_DS_ENGRAM=1` and `GR8_DS_ENGRAM_PREFIX` in `EXTRA_ENV`
([ENV.md](ENV.md)). It is part of the model, not a trade-off.

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
- Measured on `main` at b10aae4 with Q3's switches on, one boot each, two runs each (the range; the first of the two
  runs is a little slower). Short code prompts. "Each" is the median chat's writing speed once it has a slot;
  "together" is all chats; "first word" is the median wait for the first word, slot wait included.

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
    after that cached part; otherwise it reads the whole prompt again. A 1M snapshot is 882 MiB a box: about 2 fit in 2,048 MiB, about 4 in
    the 4,096 MiB default (`GR8_DS_CACHE_MIB` unset). A chat's first turn can keep 2 snapshots (prompt end and reply
    end), so this is not a fixed number of cached chats: at 2,048 MiB, only the last 1M chat stayed cached.

## 3. DeepSeek-V4.1-Flash (`ds41`): speed against exactness

Measured on four GB10 boxes (TP=4, a switch), the exlr8 release, Engram on. "Speed" is the decode rate of one chat in
tokens/s. "Quality" is a fixed 75-item check set (code, math, JSON, reasoning), the median of 3 runs. Each answer adds
words to `EXTRA_ENV` (`"A=1 B=2"`, every rank).

### Q3. Faster writing for a small quality loss?

> "Do you want DeepSeek to write faster: about 40-50% on long replies, 20% (code) to 50% (prose) on short ones? The
> cost: its output is no longer the exact model's, and our 75-item check went from 73 to 71 right."

- Default: no.
- Yes: `GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1 GR8_DS_MOE_TOPK_DECODE=4`.
- Measured against `main` at ca24fdd0 (two boots), and again on `main` at b10aae4 (one boot each way). Long replies:
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

- Default: no (each chat faster).
- Yes: `GR8_S2_ADMIT_HOLD=1 GR8_S2_TAIL_FIRST=1`.
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

1. Write `cluster.env` and show it to the person.
2. `./gr8 <model> check` and `./gr8 <model> show-config` on each box: what would stop the boot, and where each value
   came from.
3. `./gr8 <model> up` on every box, then `./gr8 <model> smoke` on the head. Each must pass.
4. `./gr8 <model> bench` on the head gives the speed of this setup ([bench/README.md](../bench/README.md)).

GLM-5.3 and GLM-5.3-Flash have no measured trade-off questions here yet.
