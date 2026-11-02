# First boot: the questions to ask

Most people will have an agent (Claude Code, Codex or another) start gr8 for them the first time. If you are that
agent, read this file before the first `up`. Ask the person the questions below, one at a time, with the numbers given
for each. Write the answers into `cluster.env` ([config/cluster.env.example](../config/cluster.env.example)), show the
person the file, and only then boot. When the person has no preference, keep the default: every default here is the
exact model.

## 1. The setup (every model)

These have no default ([DEPLOY.md](DEPLOY.md), "The two settings"):

1. **Which model?** `glm53` (GLM-5.3), `glm53f` (GLM-5.3-Flash) or `ds41` (DeepSeek-V4.1-Flash-1017). `./gr8 models`
   lists each model's release and the TP values it takes.
2. **Which boxes?** Their addresses in the same order on every box (`PEERS`; the first one is the head and serves the
   API on port 8100), or mentat with `TP`.
3. **Where do the weights come from?** A Hugging Face release into `MODEL_CACHE`, a local copy of the release, or
   model-loader ([DEPLOY.md](DEPLOY.md#with-model-loader)).

For `ds41`, also set Engram on every box: `GR8_DS_ENGRAM=1` and `GR8_DS_ENGRAM_PREFIX` in `EXTRA_ENV`
([ENV.md](ENV.md)). It is part of the model, not a trade-off.

## 2. DeepSeek-V4.1-Flash (`ds41`): speed against exactness

Measured on four GB10 boxes (TP=4, a switch), the exlr8 release, Engram on. "Speed" is the decode rate of one chat in
tokens/s. "Quality" is a fixed 75-item check set (code, math, JSON, reasoning), the median of 3 runs. Each answer adds
words to `EXTRA_ENV` (`"A=1 B=2"`, every rank).

### Q1. Faster writing for a small quality loss?

> "Do you want DeepSeek to write faster: about 40-50% on long replies, 20% (code) to 50% (prose) on short ones? The
> cost: its output is no longer the exact model's, and our 75-item check went from 73 to 71 right."

- Default: no.
- Yes: `GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1 GR8_DS_MOE_TOPK_DECODE=4`.
- Measured against `main` at ca24fdd0, two boots. Long replies: code +41%, prose +50%, structured output +23%.
  Short 256-token replies: code +21%, prose +51%. Prompt reading (prefill) does not change. Quality 73 -> 71 of 75.
- Half-way, one switch alone (one boot each, 72 of 75 for both):
  - `GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1`: long replies code +28%, prose +29%; short replies prose +20%, code
    within noise.
  - `GR8_DS_MOE_TOPK_DECODE=4`: short replies code +15%, prose +13%, structured +13%; long replies prose +17%,
    structured +16%, code within noise.

### Q2. A smaller output head?

> "Do you want about 2-4% faster writing? Some answers change (13 of 75 in our check); quality did not drop."

- Default: no.
- Yes: `GR8_DS41_HEAD_MX=1` (the decode head in MXFP8, a little over half the bytes).
- Measured: a decode round 3.7% shorter (4 boots), new-text decode +2.4%. Quality 71 -> 72 of 75; 13 of 75 answers
  changed.

### Q3. Several chats at once: first word sooner, or each chat faster?

> "When several chats start together, do you want each to show its first word sooner (about 3x with 4 chats, 5x with
> 8), if with 4 chats each one then writes about 9% slower?"

- Default: no (each chat faster).
- Yes: `GR8_S2_ADMIT_HOLD=1 GR8_S2_TAIL_FIRST=1`.
- Measured with 4 chats: median first word 648 -> 209 ms; each chat's median rate 75.7 -> 68.6 tokens/s; all 4 chats
  together +15%. With 8 chats: first word 1,656 -> 310 ms; all 8 together +5.5%. The answers did not change.
  (Measured with Q1's switches on.)

### Q4. A long prompt arrives while other chats write: keep them smooth, or answer it sooner?

> "When someone sends a very long prompt while other chats are writing, should those chats keep writing smoothly (the
> default), or should the long prompt get its first word sooner?"

- Default: smooth. The launcher sets `GR8_S2_DECODE_PF` to the smaller of 1024 and the prefill width.
- Sooner: `GR8_S2_DECODE_PF=0`. Smoother still: `GR8_S2_DECODE_PF=512`.
- Measured with 4 chats writing and a new 34K-token prompt (Q1 and Q2 switches on):

  | `GR8_S2_DECODE_PF` | each chat while the prompt reads | longest pause | the long prompt's first word |
  |---|---:|---:|---:|
  | 1024 (default) | 17.9 tokens/s | 0.49 s | 13.8 s |
  | 0 | 10.7 tokens/s | 1.15 s | 10.2 s |
  | 512 | 23.6 tokens/s | 0.36 s | 18.6 s |

## 3. Tell the person (no question)

- A `ds41` request holds at most 163,830 tokens (prompt and reply). A 34K-token prompt shows its first word in about
  9 s, a 139K-token prompt in about 37 s; writing then runs at the same speed as after a short prompt.
- Speed depends on what is asked: prose decodes slower than code.

## 4. After the answers

1. Write `cluster.env` and show it to the person.
2. `./gr8 <model> check` and `./gr8 <model> show-config` on each box: what would stop the boot, and where each value
   came from.
3. `./gr8 <model> up` on every box, then `./gr8 <model> smoke` on the head. Each must pass.
4. `./gr8 <model> bench` on the head gives the speed of this setup ([bench/README.md](../bench/README.md)).

GLM-5.3 and GLM-5.3-Flash have no measured trade-off questions here yet.
