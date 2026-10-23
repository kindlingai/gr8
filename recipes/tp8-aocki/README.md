# TP8 AOCKI at 1.31M context (8x DGX Spark, GLM-5.3)

This branch is upstream `c53b5b8` plus 79 commits:
- commits 1-39: the TP6 K3 patch series. Their tree is `25aa805615409f1375f7709ea960e200e24af13d`, the same tree as
  PR #135 (`tp6-k3-613k`) without its recipe commit. The commit ids differ from #135; the files are identical.
- commits 40-79: the TP8 work on top of it: the 8-rank hybrid fabric (route table, one relay hop, launcher `TP=8` /
  `SHAPE=hybrid`), the TP8 port onto the K3 engine, and three decode changes measured here (below).
The measured engine was built from this branch's head `a901de8` (tree `38cce08c496b31b26903f473d37a92c947003aba`).
This directory adds the serving configuration on top of it.

## What "AOCKI" is
AOCKI = arm A (the K3 settings at TP=8: MTP=3, SHARE_LIMIT 16384, CAP 1310720, `G6_CP_STEP_ROWS=8`) plus three
switches, all default off in the engine:
| switch | commit | what | effect on output |
|---|---|---|---|
| `G6_CP_ONECHAIN=1` | `c52f7a0` | context-parallel decode runs the head groups as one call at TP=8 (groups of a multiple of 8 heads) | same bits (cpgroups test, NR=8) |
| `G6_CP_STEP_KFIT=1` | `4d5bd1a` | an S2 round past R takes every session, with its drafts capped to fit the row budget | same answers at temp 0 (gate0) |
| `G6_CP_IDXLIVE=1` | `a901de8` | the batched CP indexer skips tiles past the live context and tiles of other ranks, then selects over the live chunks only | same bits (s2_attn_test cp, NR=8) |

`G6_CP_IDXLIVE` is the large one: at 4 x 200K the CP indexer was 42% of a decode step (88.7 of 211.5 ms, profiled),
because a batched step scored and radix-selected over the full capacity (1.31M) per row and ran the score math on the
7/8 of the key tiles that other ranks own.

## Files
| file | what |
|---|---|
| `envs/AOCKI.env.r0` .. `r7` | the launcher env files of the measured serve, one per rank; non-comment lines identical to the deployed files (rank 0 md5 `1aedc8378a87b1bb140f6454e219427c`). Ranks differ only in `FAB_PORTS`. |
| `handboot.sh` | boot on all eight boxes with `./gr8 glm53 up @ENV` under systemd-run, at a GPU clock cap (`CLOCK`, default 2200), wait for `/health` ready |
| `down.sh` | stop the units, `launch.py down --local` and `verify-down --kill` on each box, confirm no g6m4 is left |

`handboot.sh` and `down.sh` are cleaned-up copies of the driver that ran the serve (fleet addresses and the host
memory scrub removed). They were syntax-checked, not re-run in this form.

## Boot
Put rank k's env file on box k at the same path, then on the head box (rank 0), as root:
```
BOXES="<rank0> <rank1> ... <rank7>" H=/path/to/gr8 bash recipes/tp8-aocki/handboot.sh /path/to/AOCKI.env AOCKI
curl -s localhost:8100/v1/models    # max_model_len 1310714
BOXES="..." H=/path/to/gr8 bash recipes/tp8-aocki/down.sh /path/to/AOCKI.env
```
Boot to health: 214 s. `MODEL` is the EXLR8 GLM-5.3 release directory (`dense/`, `experts/`, `index-slices/`,
`manifests/`). Site values to change: `PEERS`, `FAB_PORTS`, `PY`, `G6_HOME`, `MODEL`; `PORT` is 8100 in the measured files.

## Gates and numbers (GPU clock capped at 2000 MHz, MTP=3)
| gate / measure | AOCKI | how |
|---|---|---|
| KV capacity | 1,310,714 tokens (max_model_len) | `/v1/models` |
| s2_attn_test `--only cp`, NR=8 | 8/8 PASS with `G6_CP_IDXLIVE` off and on (incl. 2 cases past 32768 keys) | `G6_NR=8 EXTRA=-DDEC_S2 TEST=s2_attn_test scripts/c1_tests.sh testbin --only cp` |
| gate0 (10 prompts, temp 0) vs arm A | 10/10 byte-identical | outputs compared byte for byte |
| needles, 4 x ~200K cold prompts | 4/4 found | needle at 50% of a random-word prompt |
| NLL vs the TP6 K3 serve (28,658 tokens: code, prose, tech) | 1.30108 vs 1.30102 | measured on AOCK (= AOCKI without IDXLIVE; gate0 bytes identical) |
| 8K cold prefill | 1875.5 tok/s | median of 3 |
| prose decode, short context, 1 stream | 53.0 tok/s | median of 3 |
| decode at ~200K, 1 stream | 32.0 tok/s | |
| 4 streams at ~200K each | 45.5 tok/s together (9.8-11.4 per stream) | overlap window of 4 concurrent 700-token decodes |
| host MemAvailable, rank 0, serving | 8.38 GiB | boot log |

Progress at 4 x 200K (same bench): arm A 29.4 -> +ONECHAIN/KFIT 35.95 -> +IDXLIVE 45.5 tok/s.

## Clock: 2200 MHz
`handboot.sh` caps at 2200 MHz by default. Measured on AOCK (before IDXLIVE): 2000 MHz 53.1 / 32.0 tok/s
(short / 200K, 1 stream); 2200 MHz 54.7 / 33.25 with max GPU temperature 65-71 C on seven boxes and 81 C on one;
full clock 57.0 / 35.0 with 81-87 C. 2200 keeps most of the full-clock gain at lower temperature. The gate rows above
were measured at 2000.

## KV memory: the S2 pool
    pages per rank = R/64 + (CAP - R)/(64 x TP)        R = SHARE_LIMIT = 16384, TP = 8
= 256 + 2528 = 2784 pages per rank, 10.72 GB pool storage (boot log `# s2: ... pool of 2784 pages`), about 3.9 MB/page.
`G6_CP_STEP_ROWS=16` ran out of host memory (+5.7 GiB per box); keep 8.
