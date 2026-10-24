# TP6 K3 at 613K context (6x DGX Spark, GLM-5.3)

This branch is upstream `c53b5b8` plus the 39 patches of the
[GLM-5.3-gr8-6x-DGX-Spark-TP6](https://github.com/knapcio/GLM-5.3-gr8-6x-DGX-Spark-TP6) recipe (commit `a063584`),
applied with `git am`. The resulting source tree is `25aa805615409f1375f7709ea960e200e24af13d`, the tree the measured
engine was built from (the recipe's `scripts/apply.sh` checks the same id). None of the 39 commits is on `main` yet
(`git cherry main` marks all 39 as missing). This directory adds the serving configuration we ran on top of it.

## What "K3" is
K3 = the recipe's TP6 settings with converted rank packs, plus two engine-setting changes we screened:
- K2: `G6_PF_GEMMF32=1` (byte-exact QO o_proj prefill kernel).
- K3: `G6_K1_SPLIT=1:/4,2-3:2x1,4-32:8x1/4` (the window-2 fc1 split at all multirow counts; recipe fc2 split kept).

The context comes from `CAP=612992` (9578 pages of 64 tokens, max_model_len 612986) with `SHARE_LIMIT=131072`.
At TP=6 these are close to the launcher's defaults (614400 / 131072); the earlier 131072-token serve had lowered CAP.

## Files
| file | what |
|---|---|
| `K3-cap613-sl131k.env` | the launcher env file of the measured serve; values identical to the deployed file (md5 `234b0a80...`) |
| `handboot.sh` | boot on all six boxes with `./gr8 glm53 up @ENV` under systemd-run, wait for `/health` 200 |
| `down.sh` | `./gr8 glm53 down @ENV`, then stop any unit / g6m4 left on each box |

`handboot.sh` and `down.sh` are cleaned-up copies of the scripts that ran the serve (fleet addresses and site-only
steps removed). They were syntax-checked, not re-run in this form.

## Boot
On the head box, as root, with the same gr8 checkout path `H` on all six boxes, the converted rank packs at `MODEL` and the exlr8 release at `MODEL_DIR`:
```
BOXES="<head> <box2> <box3> <box4> <box5> <box6>" H=/path/to/gr8 bash recipes/tp6-k3-613k/handboot.sh /path/to/K3-cap613-sl131k.env K3C613
curl -s localhost:8000/v1/models    # max_model_len 612986; boot log line "G6_CP on, R 131072"
BOXES="..." H=/path/to/gr8 bash recipes/tp6-k3-613k/down.sh /path/to/K3-cap613-sl131k.env
```
TP6 boots from packs: `MODEL` is the directory of converted rank packs (`rank<r>.pack` + `rank<r>.manifest.json`, plus
`qo.pack` + `qo.manifest.json` for `PF_OWN=3`; a box needs its own rank's pack, the manifests and qo.pack), and `MODEL_DIR` the exlr8 release (tokenizer and
config). Every number below was measured this way, at tag `tp6-k3-613k-measured` (dd67022). The launcher's pack boot is
#172 (it brings back what #144 removed).

The boot from the release (`MODEL` = the release) fails at TP=6: the serve reaches `/health` 200, then the first request
stops with non-finite attention at layer 37 (S2 epoch abort). This is known (seen 10-21 with the recipe engine, again
at the box gate of this PR) and tracked separately; KVB bf16 does not fix it. A TP6 release boot also needs
`index-slices/tp6` in the release: a release with tp4 slices only stops at boot with `no index slices for TP 6`.

## Numbers (GPU clock capped at 2000 MHz unless noted, MTP=3, one request unless noted)
| measure | value |
|---|---|
| boot to `/health` 200 | 24-26 s |
| 8K cold prefill | 1707 tok/s (median of 3); 1889 tok/s at full clock |
| prose decode, short context | 48.2 tok/s (median of 3); 51.0 tok/s at full clock |
| ~100K prompt, 2 sent at once | both shared the serve: 201,676 tokens in 128 s (1575 tok/s); a resend hit 98,304 cached tokens, TTFT 1.7-3.0 s |
| 245K cold prefill | 188-190 s (about 1290 tok/s), needle found |
| 245K warm turn (same prefix + new line) | TTFT 4.5 s, 241,664 of 244,649 tokens from the prefix cache |
| decode at 245K context | 29.0 tok/s |
| 2 x 245K agents at once | the second evicts the first from the pool: both re-prefill (TTFT 188 s and 393 s) |
| 50K cached turn sent during a 245K cold prefill | TTFT 9.0 s |
| host MemAvailable, lowest box | 5.14-5.34 GiB serving and under load |

## KV memory: the S2 pool
The boot log line `# s2: ... pool of N pages ... GB pool storage` gives the real KV pool per rank:

    pages per rank = R/64 + (CAP - R)/(64 x TP)        R = SHARE_LIMIT (G6_CP_R), TP = 6

The first R tokens of each session are replicated on every rank; the rest is spread over the six ranks. One page
costs about 3.83-3.93 MB per rank. A session of S tokens (prompt + max_tokens) holds R/64 + (S - R)/384 pages per
rank; sessions resident together must fit the pool, otherwise the prefix cache evicts.

| SHARE_LIMIT / CAP | pool pages | pool GB per rank | result |
|---|---|---|---|
| 131072 / 612992 (this config) | 3303 | 12.66 | serves; MemAvailable min 5.14 GiB; one 245K session (2345 pages) resident at a time |
| 262144 / 540672 | 4821 | 17.90 | out of memory during graph capture (host MemAvailable fell under 2.5 GiB on all six) |
| 49152 / 761728 | 2623 | 10.32 | serves (max_model_len 761722); two 245K sessions stay cached (warm TTFT 5.2 / 8.3 s) but a 50K turn, now past R, waited over 250 s behind a 245K cold prefill, and MemAvailable fell to 3.90 GiB |

The `attention caches + scratch ... at capacity` boot line (50-57 GB) is not resident memory. Other memory scales with
CAP as well (prefill and QO states, CP staging): going from 131072/612992 to 49152/761728 cut the pool by 2.34 GB, yet
MemAvailable fell by 0.85 GiB.

Why SHARE_LIMIT 131072: a larger R does not fit (R alone at 262144 is 4096 pages, 15.7 GB per rank), and a smaller R
puts 50K-130K agent prompts past R, where they take the context-parallel path and wait behind long prefills. With a
smaller R (65536) at the same CAP, agents with 80-130K prompts saw TTFT rise from 12-16 s to 57-90 s.
