# Flash release loader (CPU-tested)

The release's own files select the model: `hidden_size=4096` in the vocabulary file's metadata selects the Flash
path, and a release without it is GLM-5.3. Flash requires the seven source
identity fields, 46 dense layers (0–45), vocabulary, and expert widths for 3–45.
A full boot still requires the Flash runtime workstreams; this loader does not
make the existing GLM kernels Flash-capable.

## Image names and types

Flash dense weights stay lossless BF16/FP32. No MXFP8 or INT4 requantization occurs.
`source_names` and `source_shapes` restore suffixless and higher-rank tensors.
Split output rows must cover the source shape exactly, with no gaps, overlaps,
duplicate tensors, missing parts, dtype changes, or identity changes.

Layer tensors have prefix `L###.`. The aliases in `src/flash.rs:alias` retain
existing runtime names for `attn_norm`, `mlp_norm`, `q_a`, `q_a_norm`, `kv_a`,
`kv_a_norm`, `q_b`, `o_proj`, `idx.*`, `router.w`, `router.bias`, `dense.*`,
`eh`, `enorm`, `hnorm`, and `shared_head_norm`. Other tensors retain their exact
source-local names, including `.weight` when present. Examples:

- `L000.self_attn.q_proj.weight`, `L000.self_attn.q_conv1d.weight`
- `L000.self_attn.A_log`, `L000.hc_attn_fn`
- `L003.self_attn.indexer.index_kpool_compress_gate`

BF16 matrices with both dimensions divisible by 16 also get an exact `.gv`
record `[out/16,in/16,512]`. FP32 matrices remain FP32; the consumer must bind
that dtype explicitly. Norms, convolutions, and other higher-rank tensors keep
their shapes. KDA q/k/v and gate weights are currently replicated. KDA o_proj
uses the rank's 128-channel heads; DSA q_b/o_proj use 256-channel heads.
DSA kv_b becomes `kv_b.uk` and `kv_b.uv`, each `[rank_heads,256,512]`, with no
RoPE columns. Dense MLP gate/up rows and down columns use the existing rotated
intermediate ownership rule. Full-head QO prefill copies are not implemented;
requesting them fails explicitly.

Vocabulary: `G.embed` (replicated), `G.final_norm`, `G.lm_head` (rank rows), and
`G.lm_head.gv` (BF16, not the legacy MXFP8 record). Vision extras remain in the
release; this text-only image does not execute or materialize them.

Experts: H4096, E288, shared index 288, 16 halves, 65,536 bytes/half/bit.
Ownership stays `(16*expert + half + layer) % TP`. Shared K2 is supported.
The image uses the existing `tr3.*` names with 32 rather than 48 hidden blocks.

## Package and verify

Prepare explicit width manifests before this step. Use a separate destination
for a pilot. `set_id` identifies all *currently present* dense/expert weight files;
it does not certify completeness. Re-run after adding any weights.

```sh
python3 -m tools.convert.v2slices --v2 RELEASE/experts \
  --out RELEASE/index-slices --tp 4 --stamp-release RELEASE
cargo build --release --manifest-path engine/loader-rs/Cargo.toml
python3 -m unittest tools.convert.test_v2 tools.convert.test_flash_v2
python3 engine/loader-rs/tests/release_check.py --release RELEASE \
  --widths RELEASE/manifests/exlr8-manifest-k2.json --ref EXLR8_REPO \
  --dump engine/loader-rs/target/release/g6plandump --tp 4 --ranks 0,1,2,3 \
  --layers 0,3,45 --partial
python3 engine/loader-rs/tests/flash_expert_check.py --release PILOT_RELEASE \
  --ref EXLR8_REPO --work SCRATCH
```

`--partial` and `--experts-only` are **test-only** g6plandump options, not boot
options. The default `release::plan` still requires the whole model.
`flash_expert_check.py` checks every pilot half's checksum, bytes, side vectors,
and rank ownership at TP4 K2/K3/K4, plus 216 numerical half reconstructions.

For a test-only runtime subset, `m4sim_flash --v2-partial --v2-dense RELEASE`
plans only the IDs passed to `--layers`, plus globals. It uses the same
`plan_selected` path as `g6plandump --partial`. `g6m4_flash --v2-partial` does
the same on each rank (offline runs only). A normal `g6m4` or `m4sim` run still
requires the full release. This flag does not enable partial-model serving,
MTP, or S2.

The real-pilot C ABI check is `tests/partial_plan_test.cpp`: it checks all four
TP4 plans for layers 0-3, globals, expert format and a payload read. It also
checks that the normal planner rejects the incomplete pilot.

## Quantized dense records (W8)

Files marked `dense_model=flash` use the same v2 matrix codec as full GLM,
with the seven identity fields and exact `source_names` / `source_shapes`.
They are ordinary, unsplit dense/vocab files. Existing lossless split releases
remain supported. This does not make the BF16-only Flash runtime quant-capable.

With `--trellis`, format 1 K4 matrices become `.trellis`, `.suh`, `.svh`
under their existing runtime aliases. KDA q/k/v use only this rank's
128-channel heads (unlike the replicated BF16 path); KDA o uses those input
columns. DSA q_b/o use 256-channel heads. Dense MLP ownership is unchanged.
Trellis input signs for q/k/v share `attn_in` (rotation 1); dense gate/up
share `mlp_in` (rotation 3). Without `--trellis`, these files fail explicitly.

Format 3 matrices become `.mx.codes`, `.mx.scales`, `.mx.gv` without
requantization. KDA f_a/g_a, DSA q_a/kv_a, indexer wq_b/wk and MTP eh are
replicated. Format 4 kv_b becomes BF16 UK/UV `[rank_heads,256,512]` from
its dequantized values. Copied BF16/FP32 tensors keep their names and shapes;
BF16 matrices also get exact `.gv` records when dimensions permit.
Format 2 is not a Flash W8 runtime format; the loader rejects it explicitly.

Format 5 vocabulary becomes `G.embed` BF16, `G.final_norm` as shipped,
and `G.lm_head.r8.gv` + `G.lm_head.scale` (exact FP8-row E4M3 codes in tile
order plus the native FP32 per-row scales, without MXFP8 re-quantization).
The embedding and kv_b therefore round once to BF16; the head preserves its
native FP8-row weights and FP32 row scales.

### CPU tests

```sh
cargo test --manifest-path engine/loader-rs/Cargo.toml
cargo build --release --manifest-path engine/loader-rs/Cargo.toml
python3 engine/loader-rs/tests/test_flash_plan.py
```

`test_flash_plan.py` plans a synthetic Flash release with `g6plandump`: split dense tensors keep their exact bits,
and gaps, overlaps, a missing part and a model mismatch are refused.
