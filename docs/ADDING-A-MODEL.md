# Adding a model

The launcher has a declarative registry; the GPU engine is **not** a general HF
model loader. A registry entry does not make an unsupported architecture run.
The existing served models are GLM-5.3, GLM-5.3-Flash and DeepSeek-V4.1-Flash;
`pseudo` is the CPU test engine.

## 1. Engine and weights first

- Follow [HOST-CORE.md](HOST-CORE.md), especially planned steps **R6d–R11**, for
  model/module separation. Do not build a second host-core interface alongside it.
- Implement and verify geometry, weight conversion/loading and rank partitioning,
  attention operations (`engine/attn/ops.h`), module capabilities
  (`engine/module_info.h`) and S2 module/pool operations. Start with `m4sim` and
  compare outputs before attempting a distributed boot. Keep weight hash checks on.
- Declare only capabilities actually implemented. Shared page pools, per-session
  slabs and recurrent/ring state have different reuse and restore requirements.
  Cache backend availability alone does not imply model-specific SSD support.
- Compilation remains architecture-specific: inspect `engine/runtime/build_m4.sh`,
  geometry/attention selection and kernel build flags. Adding a launcher entry
  does not remove those engine-side seams.

## 2. Register once in `tools/launch/models.py`

Add one `Model` to `MODELS`. Public served-model lists derive from the registry
(`fabric=True`); the CPU pseudo model is intentionally excluded from image checks.
Use an existing entry as a checklist:

| Field | Purpose |
|---|---|
| `name`, `title`, `architecture`, `release`, `group` | CLI identity, architecture check, example release and rendezvous group |
| `engine`, `simulator`, `build` | Installed binary names and `scripts/build.sh` target |
| `build_widths`, `build_define`, `build_fc2gl` | Compile widths, model define and FC2 split |
| `build_flags_script`, `build_what_env`, `int4_row`, `build_markers` | Optional model build hooks and binary checks |
| `tp`, `variants`, `default_variant` | Supported serving widths and release variants |
| `defaults`, `tp_defaults` | Model and per-TP option defaults; CLI/config/environment override them |
| `engine_profile` | Shell defaults/env/args implementation (`glm`, `flash`, `ds41`, `pseudo`) |
| `native_release` | Use the current native-release layout rather than indexed width slices |
| `second_rail`, `per_slot_pool`, `stage_relay` | Main/prefill second rail, per-session capacity divisor, loading progress |
| `drafter`, `drafter_url`, `int4_kv`, `grammar`, `checkpoint_cache`, `kv4_fold` | Implemented optional capabilities |
| `serve_options`, `engine_prefixes`, `uniform_rows`, `vision_knob`, `min_free_gib` | Model-specific validation and supervisor behavior |
| `dialect`, `served`, `thinking_off`, `smoke_script` | Frontend protocol, API names and smoke cases |
| `vocab`, `eos_ids`, `max_positions` | Embedding bound, stop IDs and context ceiling |

`tp_defaults={'4': {'KV_FORMAT': 'int4'}}` is the single source for that default
in option parsing, shell planning and `gr8 build`; `show-config` records its origin.
Keep geometry-dependent memory calculations in the engine profile, not copied into
wrappers. If no existing profile matches, implement one in `scripts/engine_args.sh`
and route it explicitly; do not reuse GLM arguments for an incompatible engine.
`EXTRA_ARGS` reaches every model/TP, including pseudo, once.

Set `vocab=0`, `eos_ids=()` and `max_positions=0` to read release metadata. Explicit
fields win; otherwise `config.json` supplies vocabulary/context, including nested
VLM `text_config`. EOS uses `generation_config.json` before `config.json`, scalar
or list, deduplicated without unioning overridden stop lists. Unknown models with
missing vocabulary/EOS fail rather than silently using GLM constants. The shim
also accepts `--model-dir` for standalone discovery, with explicit flags taking
precedence. Existing GLM/pseudo hello defaults are retained by `shim_legacy`;
new models should leave it false. `legacy_identity` is only for the original
GLM handshake digest, not new registrations.

On a checkout, `gr8 <model> build` reads the registry; newly registered build
targets fall back to its `build-shell` query. Shipped Dockerfile engine targets
(`glm`, `flash`, `ds41`) instead use pure-shell metadata in `scripts/build.sh`:
`engine-src` needs neither Python nor `tools/launch/models.py`, so launcher edits
do not invalidate CUDA compile layers. Keep that shell table in sync when changing
served build metadata; `tools/launch/test_registry.py` enforces full parity.

Other shell consumers query fields without duplicating model-name lists:

```sh
python3 tools/launch/launch.py model-get glm53 build
python3 tools/launch/launch.py model-get glm53 default-kv-format 4
python3 tools/launch/models.py get image-binaries
```

The `native_release` path currently uses the native layout helpers in
`tools/launch/modelset.py`; audit tensor names, manifest identities and required
files for your architecture. V2 releases similarly need loader/HF-fetch planning
and tokenizer-file lists audited, especially when the template is embedded in
`tokenizer_config.json` instead of a separate file.

## 3. Frontend

Use `dialect='hf'` with the generic HF frontend change for text templates emitting
`<think>...</think>` and JSON `<tool_call>{"name":"f","arguments":{}}</tool_call>`.
That frontend accepts `chat_template.jinja` or the `default` template in
`tokenizer_config.json`. It does not parse arbitrary tool syntax such as
`[TOOL_CALLS]`. A custom dialect needs explicit renderer/parser integration and
streaming chunk-boundary tests. See [frontend/README.md](../frontend/README.md).
This launcher change itself does not add the `hf` Rust dialect.

## 4. Packaging and operations

- Add the engine compile/dist stages and matrix legs in `Dockerfile` and
  `.github/workflows/ci.yml`. Runtime image binary checks and every model/TP
  `show-config` smoke now derive from the registry; a missing Flash binary or
  `kvman-remote` fails the image gate. These checks do not compile a new engine.
- Set `smoke_script` to appropriate assertions; generic smoke cases still assume
  thinking/tool capabilities. Add model-specific cases rather than weakening them.
- `gr8 switch` accepts any configured subset of at least two registered GPU models.
  Supply each one's `NAME_BOXES`, `NAME_DIR`, `NAME_HEALTH_URL` plus global `BOXES`
  in a literal config file. Partial declarations and unconfigured targets fail.
- Audit console presets (`tools/console/server/presets.py`), recipe metadata,
  knob descriptions and model-specific docs separately. They are not automatically
  generated from registry metadata.

## 5. Prove and review

```sh
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s tools/launch
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s tools/switch
CUDA_VISIBLE_DEVICES='' bash .github/ci/cpu-tests.sh
```

CPU tests cover registry queries, metadata precedence, switch subsets, command
parity for existing model/TP/KV variants and `EXTRA_ARGS`. Build every intended
CUDA topology on the target hardware, then run the model's smoke tests and
`bench --baseline` under identical boot conditions. Record exact output/numerical
comparisons, context and memory headroom; CPU tests do not establish GPU
correctness or speed. Follow [TESTING.md](TESTING.md) and the repository review
and release gates before shipping.
