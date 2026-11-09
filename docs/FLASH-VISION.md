# GLM-5.3-Flash image input (`flash`)

The `flash` engine takes images in chat requests (issue #407), the way the checkpoint's own code reads them
(transformers `models/glm5_next`, vLLM `vllm/models/glm5next/common/multimodal.py`). It is off by default
(`GR8_GLM53F_VISION`). DeepSeek-V4.1-Flash got image input first ([DS41-VISION.md](DS41-VISION.md)); this one shares
its protocol, admission checks, digest, refusals and rank agreement.

## What the reference does

These are the facts the engine and the front end follow. Sources: transformers 5.22 (`image_processing_glm5_next.py`,
`modeling_glm5_next.py`, `processing_glm5_next.py`), vLLM main of 2026-11-08 (`vllm/models/glm5next/common/`), the
release's `config.json` and `processor_config.json`.

### Positions

- The text model has no RoPE anywhere. `qk_rope_head_dim` is 0 and `rope_parameters` is null, so the MLA attention is
  NoPE (`mla_use_nope`), and the sparse indexer's rotary part is empty too: vLLM's indexer sets
  `rope_dim = qk_rope_head_dim` and skips the rotary when it is 0 (`indexer_rope_interleave` only picks the layout
  of a rotary that is not there). The KDA layers have no positions.
- vLLM uses M-RoPE positions only when the text config's `rope_parameters` has an `mrope_section`, and this one has
  none, so image tokens get the ordinary positions 0, 1, 2, ... like text. A position only picks a KV row and bounds
  the causal mask (`flash` already works this way, [FLASH.md](FLASH.md)).
- So an image token differs from a text token only by its input embedding: the tower's output row replaces the
  `<|image|>` embedding (`masked_scatter`), and nothing else in the text model changes.

### Prompt

- The chat template renders an `image` or `image_url` content part as `<|begin_of_image|><|image|><|end_of_image|>`
  (154830, 154854, 154831), in any message (system, user, assistant) and in tool results.
- The processor grows the one `<|image|>` to `grid_h * grid_w / 4` copies of it (`merge_size` 2), where the grid is the
  padded image in 14 px patches. The begin and end tokens stay single text tokens.
- Video (154832, 154855, 154833) is out of scope.

### Processor (`Glm5NextImageProcessor`)

`processor_config.json`: patch 14, merge 2, temporal patch 2, `patch_expand_factor` 1, `min_image_tokens` 16,
`max_image_tokens` 8000, CLIP mean and std, BICUBIC.

1. `smart_resize(num_frames=2, h, w, factor=28, min 16, max 8000 tokens)`: the target canvas is `h` and `w` rounded up
   to multiples of 28. A canvas below 16 tokens is scaled up to it. A binary search on the height cuts a canvas above
   8000 tokens down, keeping the aspect.
2. The content size: `scale = min(target_h / h, target_w / w)`, capped at 1 unless the image is below the token floor,
   then `floor(h * scale)`, `floor(w * scale)`. An image inside the budget is not resampled at all: only one below
   the floor (under about 12.5K pixels) or above the cap (over about 6.27M pixels) is.
3. The content is resized (BICUBIC) when its size changed, then padded with zeros (black) on the right and the bottom to
   the target canvas. There is no centring and no grey fill.
4. `x = (u - fl32(mean * 255)) / fl32(std * 255)` in float32 (the torchvision backend fuses the rescale into the
   normalize). The PIL backend computes `u / 255` first and is at most one float32 ulp away.
5. Patches: 2 x 2 merge blocks in row-major order, the 4 patches of a block in row-major order, each patch
   `(channel, frame, y, x)` with the one frame repeated over the 2 temporal slots (1176 values).

The default backend is torchvision. Its BICUBIC is not PIL's: on the resampled images in
`engine/ref/flash/fixtures/vision/pixels.json` it is up to 2 levels away from PIL on 0.01% to 0.3% of the values.
Inside the budget the two backends give the same pixels. The front end follows the PIL backend, which it can match
byte for byte.

### Tower (`glm5_next_vision`)

`vision_config`: depth 24, hidden 1024, 16 heads (head 64), MLP 4096, `out_hidden_size` 4096 (the text hidden size),
`projection_intermediate_size` 10240, `swiglu_limit` 10, RMSNorm eps 1e-5. 347 BF16 tensors, `model.visual.*`.

1. Patch embed: Conv3d (2, 14, 14) with bias, as a 1176 -> 1024 linear.
2. 24 blocks: `x += proj(attn(norm1(x)))`, `x += mlp(norm2(x))`.
   - attention: `qkv` (bias) 1024 -> 3072, per-head RMSNorm of q and k (weights of 64), a 2-D rotary on the whole head
     (16 frequencies `1 / 10000^(2i / 32)`: angle `j < 16` is the patch row times `f_j`, `16 <= j < 32` the patch column
     times `f_(j-16)`, `x1 = x[:32]`, `x2 = x[32:]`), full (non-causal) attention over the image's patches, scale 1/8,
     `proj` (bias).
   - MLP: `down(silu(min(gate, 10)) * clamp(up, -10, 10))`, all three with bias.
3. `post_layernorm` (RMSNorm), then each 2 x 2 block of patches (adjacent in the patch order) goes through `downsample`,
   a Conv2d (2, 2) with bias, as a 4096 -> 4096 linear.
4. Merger: `proj` 4096 -> 4096 (no bias), LayerNorm (eps 1e-5, with bias), exact GELU, then
   `down(silu(min(gate, 10)) * clamp(up, -10, 10))` with 4096 -> 10240 -> 4096 (no bias).

The output is one 4096-wide row per image token, in the patch order's 2 x 2 blocks, which is the row-major order of
the merged grid.

vLLM and transformers differ in two small ways: vLLM's block norms use eps 1e-6 (transformers: the config's 1e-5),
and vLLM's RMSNorm multiplies the weight in float32 before it rounds. The engine follows transformers.

### What the golden data shows

`engine/ref/flash/vision_golden.py` runs transformers' tower on the CPU in float32 and in bf16. The bf16 run is 5% to
11% (relative Frobenius) away from the float32 run, with a median row error of 3% to 5%. The large part comes from a
few patches: in flat regions the tower turns some patches into massive-activation tokens (norms near 800 against
35), and which of several near-identical patches takes that role depends on the rounding. Any two runs that round
differently disagree on those rows, so the tests judge the median row error and the whole output's error, each
against the bf16 run's own.

## What a request does

1. Front end (`frontend/g6front-rs`). A chat request whose messages hold an `image`, `image_url` or `input_image`
   part renders through the checkpoint's template (`template.rs` `render_images`). The template places each part
   where it likes (it sorts tool results into their calls' order), so the front end asks it: with n parts it renders
   ceil(log2 n) more times, marking part i as an image when bit b of i is set and as a video otherwise, and reads each
   `<|image|>` placeholder's part off those bits. One more render with every part a video finds a literal
   `<|image|>` in the text, which is refused. Only inline data is accepted, as for DS41 (a data URL, an Anthropic
   base64 `source`, or `data`). Each image is decoded and padded (`vision.rs` `glm5v`: smart_resize, Pillow's
   BICUBIC when the content size changes, then zeros to the canvas, byte for byte against the PIL backend on lossless
   input), and each placeholder grows to `vit_h * vit_w / 4` copies of `<|image|>`. The generate frame's `images` carry the padded
   RGB (PROTOCOL.md). Limits: 16 images and 48 MiB of base64 a request (shared with DS41).
2. Shim (`frontend/g6s2shim.py`). The head writes `DIR/vision.s2` holding `glm5v`, so the shim checks the Flash
   grids and token (DS41's file holds `1`).
3. Engine, admission (`engine/runtime/image_input.h`, shared with DS41). `image_model()` is GLM5V_IMAGES in the
   Flash build: token 154854, and a grid is valid when both sides are even and it has 16 to 8000 tokens. The pixel
   digest joins the request's cache identity, so an image prompt shares Flash checkpoints only with a prompt with the
   same images at the same positions. A request is refused (`end error`) when the tower is not loaded, when it asks
   for prompt logprobs (the prompt then runs as decode steps), or when the prompt ends inside a span.
4. Engine, prefill (`flash_native.cuh` `on_prefill`, `flash_vision_stage.cuh`). Before each prefill launch,
   every image that overlaps the launch's rows runs through the tower (`flash_vision.cuh`), and its overlapping rows
   are copied to a device pool. The embed (`flash_prefill.cuh` `pf_embed_flash`, `mhc.cuh` `k_mhc_embed_rows`) then
   writes those rows into the 4 mHC streams in place of the `<|image|>` embedding. The tower's last output stays in
   its staging buffer, so an image that spans two launches (two lanes, or two quanta) runs once. A launch without
   image rows takes the text-only embed kernel. Decode steps never hold an image row: the last prompt token is a
   decode row, and the template's `<|end_of_image|>` follows every span.

The drafters (MTP and DFlash2) get no vision input of their own. The MTP layer embeds the `<|image|>` token at an image
row, and DFlash2 reads the main model's taps (which do carry the images) but embeds its own inputs from the token ids.
With exact acceptance this changes only how many drafts are accepted, and the output stays the same.

## Knob

`GR8_GLM53F_VISION` (agree, default 0). With `GR8_GLM53F_VISION=1`, `--serve`, `--prefill` and a release (`--v2-dense`)
that has `dense/vision-bf16.safetensors`, every rank loads the tower at boot. The boot line is
`# rank R: flash vision: tower ..., N MiB on the device (...)`. The launcher (`./gr8 glm53f`) fetches the file only
when `--extra-env` sets the knob (hfsource.py, mlsource.py), and a rank fetched before that is not ready until it has
it. 0 never loads the tower and the engine runs the text-only code. A rank that loaded the tower adds
`vision <file id>` to the rank agreement (vision_id.h, as DS41), so ranks that differ refuse at bootstrap.

## Costs and limits

- Device memory on every rank: the weights (1,075 MiB), the workspace (473 MiB, sized for the largest image of 8000
  tokens and 32000 patches, with the MLPs in row passes), a pool of one quantum's rows (`MP * lanes * 8 KiB`) and one
  image's staging (64 MiB): 1,643 MiB at `--prefill 4096` on one lane, 1,675 MiB on two. The boot counts them before
  the ready check, so a box at its `GR8_MEM_MIN` floor needs a smaller `CAP` with vision on.
  - One GB10 with the one-box recipe's settings (int4 KV, `SESS=4`, DFlash2, `GR8_MEM_MIN=5`): `CAP=700032` stops at
    4.81 GiB left with vision on. `CAP=524288` boots.
  - Two GB10s with the two-box recipe's defaults (bit349, int4 KV, `SESS=8`, `CAP=2097152`, DFlash2): the pair boots
    with vision on, with 12.9 / 13.8 GiB available at ready against 14.2 / 14.1 GiB with the knob at 0.
- The tower runs on the serve thread, before the launch: a request with images holds the other sessions' steps for
  its tower time, about 3 s for an 8000-token image.
- Tower time on one GB10 (fused attention, cuBLAS linears): about 12 ms for a 63-token image, 143 ms for 1024 tokens,
  1.05 s for 4096 tokens and 3.2 s for 8000 tokens. Attention over all of an image's patches grows with the square of
  its size.
- The front end follows the PIL backend. The torchvision backend (transformers' default) resamples up to 2 levels
  away on the few images that need a resize.
- Video is not supported.

## Tests

No GPU:
- `cd frontend/g6front-rs && cargo test --release`: `glm5v_plan_matches_the_processor` (the grid for 24 sizes),
  `glm5v_pixels_match_the_processor` (the three tower fixtures and 6 resized images against the PIL backend's bytes),
  `render_images_follows_the_template`; with `GR8_TEST_TOKENIZER_DIR=<release>`, `render_images_in_the_flash_template`
  (the real template, with sorted tool results).
- `python3 -m unittest frontend/tests/test_s2shim.py`: the shim's Flash grids.
- `scripts/host_tests.sh`: `image_request_test` (the Flash geometry beside DS41's) and `agree_vision_test` (both models'
  rank agreement on loopback).
- `tools/launch`: `test_hf.py` and `test_mlsource.py` `test_the_vision_tower_only_with_image_input`.

One box (TP=1, the one-box recipe's settings with `CAP=524288`): `./gr8 glm53f smoke` passes 14 of 14 with the knob at 0
and at 1. With the knob at 1 the model names the red disc, reads the bar chart's four labels and values, describes the
scene, answers about two images in one request, and describes a 2700 x 2600 image (7,917 tokens, one tower run across
two prefill quanta), alone and beside four text sessions. Text replies at temperature 0 from fresh boots with the knob at
0 and at 1 are the same.

Two boxes (TP=2 on one direct 200G cable, the two-box recipe's settings): smoke passes 14 of 14 with the knob at 0 and at
1, the same image answers come back, and the 7,917-token image runs once across both prefill lanes. A rank whose tower
file has another header refuses at bootstrap, and the launcher refuses a knob set on one box only.

One GPU: `engine/runtime/tests/flash_vision_test.cu` (its header has the command) on the release's
`dense/vision-bf16.safetensors` and `engine/ref/flash/fixtures/vision`. Pass = projected error and median row error
within 1.5x the reference bf16 run's own, repeat runs identical. It prints the tower time for 1024, 4096 and 8000
tokens.
