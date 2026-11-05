# DeepSeek-V4.1-Flash image input (`ds41`)

The `ds41` engine takes images in chat requests, the way the release's own code reads them
(`inference/vision.py`, `inference/image_processor.py`, `encoding/encoding_dsv4.py`).

## What a request does

1. **Front end** (`frontend/g6front-rs`). An `image`, `image_url` or `input_image` block in any message's content
   list, or in a tool result, renders as one `<｜deepseek_image｜>` in the prompt (`src/dschat.rs`; golden against the
   official encoder, `src/ds41_vision_golden.json`). Only inline data is accepted (a data URL, an Anthropic base64
   `source`, or `data`). A URL the server would have to fetch is refused. Each image is decoded, resized and padded
   (`src/vision.rs`, `plan_image_grid` and `ImageOps.pad`; bit-exact to PIL on lossless input), and the one token
   becomes the image's span of `llm_h * (llm_w + 1) + 2` image tokens. The front end sends the padded RGB to the engine in
   the `generate` frame's `images` (PROTOCOL.md). Limits: 16 images and 48 MiB of base64 a request.
2. **Engine, admission** (`engine/runtime/ds41_image.h`, `s2/sessions.h`, `s2/service.h`). The shim's `img` header
   fields are parsed and checked against the prompt (every span row is the image token, spans in order, no overlap).
   The pixel digest joins the request's cache identity, so an image prompt never shares prefix-cache pages with a
   text prompt or with a prompt that has other images (control protocol version 14). A request is refused
   (`end error`) when the tower is not loaded, when the prompt would not prefill (prompt_logprobs, steering, or
   `GR8_DS_PREFILL_MIN` 0 / a prompt below it), or when the prompt ends inside a span.
3. **Engine, prefill** (`g6m4.cu` `dc.prefill`, `prefill2`, `prefill_pack`; `ds41_step.cuh` `Ds41Vision`). Before each
   eager prefill launch, every image that overlaps the launch's rows runs through the tower and aligner
   (`ds41_vision.cuh`), and its overlapping span rows are copied to a device pool. The embed op then writes those rows in
   place of the image token's embedding, in all 4 mHC streams. The same rows are flagged for:
   - the router: image rows select experts with `gate.bias_vl` instead of `gate.bias` (main layers; `kernels/ds_router.cuh`);
   - Engram: an image token is dead in every n-gram (the window reads -1, so the hash pads from there on), and the
     gate is 0 on image rows (`runtime/ds41_engram_ops.cuh`, `engram/engram_dev.cuh`). This is official `engram.py`.
   Captured graphs never hold image rows: the last prompt token is a decode row and is never in a span.

## Knob

`GR8_DS41_VISION` (agree, default 0: image input is opt-in). With `GR8_DS41_VISION=1`, `--serve`, `--v2-dense` and `GR8_DS_PREFILL_MIN` set, a release that has
`exlr8-extra-vision.safetensors` (in the dense directory or in `dense/`) loads the tower at boot. The head then writes
`DIR/vision.s2`, and the shim's hello offers `vision`. The boot line is
`# rank R: ds41 vision: tower ..., N MiB on the device (...)`. `0` (the default) never loads the tower, and the engine runs the
text-only code (the router and Engram take the text-only kernels: their image arguments are null).
Each rank looks for the tower file on its own. A rank that loaded it adds `vision <file id>` (the file's byte size and
a hash of its safetensors header) to the rank agreement; a rank without it adds nothing. So ranks that differ (a box
without the file, or with another file) refuse at bootstrap.

A prefix-cache entry and a `GR8_DS_PARK` resume rebuild the Engram ring with -1 at the request's image rows, as the
live steps leave it (`engine/runtime/s2/ds41_engram_ring.h`).

## Costs and limits

- Device memory: the tower weights (about 1.1 GB) + its workspace (about 0.47 GB) + a span pool of `2 * W` rows
  (`W * 20 KiB`), on every rank.
- The tower runs on the step loop's thread, before the launch: a request with images holds every other session's
  steps for the tower time of its images (first follow-up: run it on a side stream at admission).
- An image that straddles two prefill chunks is encoded once for each chunk.
- JPEG and lossy WebP decode a few levels from PIL (another decoder); lossless input is exact.
- The DSpark (MTP) draft layers get no vision input of their own: their embedding at image rows is the image
  token's. Not checked against the official MTP code. With strict (exact) acceptance it can only change how many
  drafts are accepted, not the output. With relaxed acceptance (`GR8_DS_RELAXED_ACCEPT`) a draft token is kept
  when it is close enough to the main model's, so a draft image embedding can change the emitted token.

## Tests (no GPU)

- `cd frontend/g6front-rs && cargo test --release`: the encoder golden, the PIL comparison, span expansion,
  the wire format. `GR8_TEST_DS41_TOKENIZER_DIR=<release dir>` adds the real-tokenizer check (the placeholder is one
  token, 129264).
- `python3 -m unittest frontend/tests/test_s2shim.py`: the shim's `img` fields and their checks.
- `scripts/host_tests.sh`: `image_request_test` (the parse, the checks, the digest in the cache identity and the
  control frames, the refusals).
- `cargo test --release` in `engine/loader-rs`: `router.bias_vl` bound on the main layers only.
- `engine/ref/ds41/vision_layout_check.py`: a CPU replay of every cuBLAS call and kernel index of the tower.
- `scripts/host_tests.sh` also runs `ds41_engram_ring_test` (an image 0..3 tokens before a prompt-end or request-end
  snapshot: the rebuilt ring equals the live ring) and `ds41_agree_vision_test` (4 ranks on loopback: a rank without
  the tower file, or with another file, refuses).

## Boot test (GPU; one TP4 `ds41` serve)

1. Tower alone, one GPU: build and run `engine/runtime/tests/ds41_vision_test.cu` (its header has the command) on
   the release's `exlr8-extra-vision.safetensors` and `engine/ref/ds41/fixtures/vision`. Pass = delimiters exact,
   aligner rows within tolerance on the 3 images, repeat runs identical; it prints the tower time per grid.
2. No image rebuild: every boot plans the rank image from the release and loads it (engine/runtime/packfind.h
   rs_plan_release; no on-disk cache for ds41), so the new binary binds `router.bias_vl` itself. Boot the usual serve twice,
   with `GR8_DS41_VISION` unset (0) and with `GR8_DS41_VISION=1`. Compare the ready memory per rank (the `ds41 vision:` boot line has
   the device bytes). With 1, `DIR/vision.s2` exists and the front end's hello has `vision`.
3. Text unchanged: the usual text battery against both boots and against the current serve reference. With the
   same seed and temperature 0 the outputs must match the reference token for token (the text path takes the
   same kernels).
4. Images: send the fixture PNGs (`engine/ref/ds41/fixtures/vision/*.png`) as data URLs, one at a time and all 3 in
   one request, in `/v1/chat/completions`:
   `{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
   {"type": "text", "text": "Describe this image."}]}], "max_tokens": 200, "temperature": 0}`.
   Pass = answers that describe the image, a `# ds41 vision: slot S image HxW patches ... ms` log line per image
   and chunk, no engine error. Also: an image prompt shorter than `GR8_DS_PREFILL_MIN` gets `end error` and the
   engine keeps serving; an image request with `GR8_DS41_VISION=0` gets an API error.
5. Concurrency: image requests alongside text sessions (the 32-slot serve). Watch the text sessions' step gap while
   a tower runs.
