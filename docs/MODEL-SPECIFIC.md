# What is GLM 5.3-specific, and what is not

g6 was written for one model, GLM 5.3 (full), on one topology (six GB10 ranks). Nothing here is a plug-in model
interface. This page is a map for porting: it lists every place the model is baked in, roughly in the order you would
meet them. It also lists the parts that do not depend on the model.

GLM 5.3 as g6 sees it: hidden 6144; 78 layers, of which 0-2 are dense MLP (intermediate 12288) and 3-77 are MoE with 256
routed experts, top-8, expert intermediate 2048, plus one shared expert (2048); one MTP layer (index 78); MLA attention
with 64 heads, q_lora 2048, kv_lora 512, qk 192 + 64 rope, v 256; a sparse indexer (32 heads x 128, top-2048 keys) on
the full-index layers; vocab 154,880.

## Model-specific: change these for another model

| layer | where | what is fixed |
|---|---|---|
| weight format | `engine/kernels/tr3.cuh`, `tr3_gemm.cuh`, `tr3_wide.cuh` | routed experts are EXL3 trellis codes (3.25 bpw, Hadamard-128 rotations, K3/K4 per-expert bit widths), decoded inside the expert kernels. Two encodes with one layout: TR3, and exlr8-e4m3, whose codebook values are rounded to FP8 E4M3 (decode rounds them too, chosen per layer from the pack manifest) |
| | `engine/kernels/densev.cuh`, `gemm_x.cuh`, `rowtile.cuh` | dense weights: MXFP8 for prefill + lm_head, GPTQ int4 g32 for decode |
| converter | `tools/convert/plan.py` | `H = 6144`, 78 + 1 layers, 256 experts, the rank map: TP6 head groups `(11,11,11,11,10,10)`, expert 128-column blocks `(3,3,3,3,2,2)`, vocab tiles, rotated per layer (TP4: 16 heads, 4 blocks, dense 3072 channels, vocab tiles `(303,303,302,302)`) |
| | `tools/convert/convert.py` | checkpoint tensor names (`model.layers.N.self_attn.q_a_proj.weight`, ...), MLA reshapes (`q_b` -> 64 x 256 x 2048, `kv_b` -> 64 x 448 x 512), dense layers 0-2, the MTP layer's extra tensors (`eh`) |
| | `tools/convert/qopack.py` | `qo_edges` head-group edges for the full-head QO pack (64 heads) |
| | `engine/ref/g6ref.py` | record layouts the kernels read (rt_record, GEMV records) |
| kernels | `engine/kernels/common.cuh` | `HID = 6144` |
| | `engine/kernels/k1args.cuh` | `SMAX = 72` distinct experts per step, `ACTW` routed columns per rank (`g6nr::EB * 128`), `SACTW` shared/dense columns (`g6nr::DENSE_CH`) |
| | `engine/kernels/glue.cuh`, `k1.cuh` | router (sigmoid top-8 of 256 + shared), RMSNorm, embed, head local argmax/LSE + sampling |
| attention | `engine/attn/` | MLA latent attention (kv_a 576 = 512 + 64 rope, UK/UV absorb), the DSA-style sparse indexer and its exact top-2048 selector (`select.cuh`), value head dim 256 in decode (`decode-small.cuh`) |
| | `engine/runtime/rank.cuh` | `QA = 2048`, `KVA = 576`, `IDXQ = 4096`, `IDXK = 128`, `IDXW = 32`; the layer sequence (dense / MoE / MTP) |
| | `engine/runtime/qo_weights.cuh`, `attn/prefill-qo-*` | full-head QO attention shapes |
| drafter | `engine/runtime/dflash.cuh`, `tools/convert/dflash2.py` | the GLM 5.3 DFlash2 drafter: 6 layers, 64 q / 8 KV heads of 128, MLP 12288, block 8, window 2048, two-tap grouped convs (group 16), target layers 5..75 (from the pack), selector rank 256 / top 16; TP must divide the 8 KV heads |
| front end | `frontend/g6shim.py` | `EOS_IDS = [154820, 154827, 154829]`, `VOCAB = 154880` |
| | `frontend/g6front/glm_parser.py` | GLM's `<think>` reasoning and `<tool_call>NAME<arg_key>..</arg_key><arg_value>..</arg_value></tool_call>` format |
| | `frontend/g6front/template.py`, `tokenizer.py` | read `chat_template.jinja` / `tokenizer.json` from `--model-dir`, so these follow the model directory |

## Topology-specific (TP6, TP4)

- `engine/g6nr.h`: the rank count `G6_NR`. Default 6; `-DG6_NR=4` builds TP4 (`G6_NR=4 scripts/build.sh`). Every
  per-rank width the engine uses (heads, expert blocks, dense channels, vocab slice, scratch sizes) comes from it;
  the NR=6 values equal the old TP6 literals, so the TP6 build is SASS-identical. The engine also compiles for other
  divisors of 64, but only 6 and 4 have been run.
- `tools/convert/plan.py`: `set_nr(n)` picks the widths; they split n units the way `g6nr.h` does (larger chunks first).
- `tools/convert/convert.py --d8`: one GPTQ dir per rank (six for TP6, four for TP4), because the GPTQ codes are solved
  per rank slice. The count of dirs sets the rank count of the packs. TP4 needs the TP4 codes.
- `G6_PF_OWN=4` (`engine/runtime/qp.cuh`, `qp_wire.h`, the pair code in `qo_weights.cuh`, `tools/convert/pairpack.py`):
  TP4 only. It assumes four ranks in two head pairs of 16 heads each (64 heads, head group of rank m in layer l =
  (m - l) mod 4) and the q_b / o_proj / kv_b shapes of GLM 5.3.
- `scripts/launch_tp6.sh` / `scripts/launch_tp4.sh`: the rank count and the KV pool. `stage.sh` and `stop.sh` work
  for any number of `HOSTS`.

## Not model-specific: reusable as is

| part | where |
|---|---|
| RoCE v2 verbs fabric: device discovery by IP, two rails, all-reduce / reduce-scatter / all-gather with device doorbells, bulk proxy | `engine/comms/` |
| pack format (sequential tensors, 4096-byte alignment, sha256 per tensor in a JSON manifest), writer, checker | `tools/convert/pack.py`, `checkpack.py`, `engine/loader/` |
| device image loader (whole pack into one allocation, optional hash check) | `engine/loader/devimage.*` |
| serve loop (file protocol), step plan, CUDA graph capture per row count, MTP draft/verify loop | `engine/runtime/serve.h`, `rank.cuh` |
| context-parallel KV layout and the staged-page hash check | `engine/comms/cplayout.h`, `engine/runtime/prefill.cuh` (CP sections) |
| OpenAI/vLLM-compatible HTTP server, metrics, streaming, socket protocol | `frontend/g6front/`, `PROTOCOL.md` |
| launch / stage / stop scripts | `scripts/` |

## A realistic porting path

1. Start with a model whose blocks match: MLA + MoE, ideally the same EXL3 trellis format. Then most changes are the
   constants above and the converter's tensor names.
2. Keep `m4sim` as the test bed. It is one GPU simulating all six ranks on a few layers (`--layers 0,3,40,77`), and
   `scripts/build.sh` builds it. Every rank's step lines must match the six-rank run on the same packs. That is how we
   checked each kernel change.
3. A different attention type (GQA, no indexer) or a different expert format is new kernel work in `engine/attn/` or
   `engine/kernels/`, not a configuration change.
