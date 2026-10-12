# PROTOCOL — gr8 front end <-> engine (v1)

The front end (`frontend/g6front-rs`) owns HTTP, tokenization, chat templates, detokenization, stop strings,
reasoning/tool parsing and the OpenAI/vLLM response shapes. The engine (C++/CUDA, rank 0) sees **token ids only**.
They talk over one Unix stream socket. The engine listens on it; the front end connects.

Default path: `/run/g6/engine.sock` (both sides take a flag to change it).

## Framing

Every message in both directions is one frame:

    u32 length, big-endian  |  `length` bytes of UTF-8 JSON (one object)

- Max frame 256 MiB (a 200K-token `prompt_logprobs` result with k=20 is ~100 MB of JSON; everything else is small).
- Every object has a `"type"` string. Unknown types are ignored by both sides; unknown fields are ignored.
- JSON numbers: token ids are integers; logprobs are floats. `-inf` is not valid JSON: send `-9999.0` instead.
- One connection carries many requests, multiplexed by `"id"` (a string chosen by the front end, unique while in
  flight). Frames for one request arrive in order; frames of different requests may interleave.
- If the connection drops, every in-flight request is dead on both sides. The engine aborts them and frees
  their KV (the prefix cache may keep blocks). The front end fails them with HTTP 500 / a stream error chunk
  and reconnects.

## Engine -> front end: `hello` (first frame after accept)

```json
{"type": "hello", "version": 1, "model": "glm-5.3", "max_model_len": 200000, "vocab_size": 154880,
 "eos_ids": [154820, 154827, 154829], "max_logprobs": 20}
```

The front end refuses to serve if `version` != 1. `max_model_len` bounds prompt + max_tokens. `max_logprobs` is
the largest k the engine accepts for `logprobs` and `prompt_logprobs`. `"insert": true` (absent = false) says the
engine applies `generate`'s `insert` rule.

An optional `steering` object advertises a loaded profile: `profile` (name), `digest` (64 lowercase hex
manifest SHA-256), `scope` (`shared` or `full`), `layers` (positive active-layer count), `forced` (boolean),
and `forced_scale` (canonical FP32 number; zero unless forced). Absent/null means unavailable. Current real
engines advertise no loaded profile; the synthetic mock's `--steering` option exercises transport only.
See [request transport](docs/STEERING-TRANSPORT.md).

## Front end -> engine

### `generate`

```json
{"type": "generate", "id": "cmpl-3f2a...",
 "prompt": [154822, 154824, 154826, 25062],
 "max_tokens": 512, "min_tokens": 0,
 "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": null,
 "stop_ids": [154820, 154827, 154829],
 "logprobs": null, "prompt_logprobs": null}
```

| field | meaning |
|---|---|
| `prompt` | prompt token ids, already templated (the engine never sees text) |
| `max_tokens` | hard cap on generated tokens (>= 1). The front end has already clamped it to `max_model_len - len(prompt)` |
| `min_tokens` | the engine must not stop on `stop_ids` before this many tokens (0 = off) |
| `temperature` | 0 = greedy (argmax; `top_p/top_k/min_p/seed` then have no effect) |
| `top_p` | nucleus mass, 1.0 = off |
| `top_k` | 0 = off (the front end maps OpenAI/vLLM `-1` to 0) |
| `min_p` | 0.0 = off; keep tokens with p >= min_p * p_max |
| `seed` | integer or null. With a seed the sampled sequence is a pure function of (seed, logits) |
| `stop_ids` | generation stops after emitting any of these ids. The front end always sends the full list (EOS ids from `generation_config.json` + request `stop_token_ids`), empty if `ignore_eos` |
| `logprobs` | null = none. k >= 0: for every generated token return its logprob and rank plus the top-k (k = 0 means the sampled token only) |
| `prompt_logprobs` | null = none. k >= 0: for every prompt position j >= 1 return the logprob and rank of `prompt[j]` under the distribution predicted at j-1, plus the top-k |
| `system_tokens` | optional (chat requests). How many of the prompt's first tokens are its system span: the leading system messages and the tool definitions as the chat template renders them, up to where the first other message begins. The engine's KV stores (`--kv-keep system`) keep system prefixes by it. Absent (completions): the engine counts every prefix of the prompt instead |
| `steering_profile` | optional profile name; must match the engine's loaded capability, even when scale is zero. Missing/null selects stock unless the engine is in forced mode |
| `steering_scale` | optional finite number in [0,4], default 1 with a profile. Converted to FP32 once, negative zero normalized to positive zero. A nonzero scale requires a profile; zero selects the stock identity |
| `insert` | optional, only to an engine whose hello says `"insert": true`. A list of at most one rule `{"after_tokens": N, "unless_emitted": X or null, "tokens": [ids]}` (1 to 64 ids): once the reply holds N tokens and none of them is X, the engine puts `tokens` in place of the reply's next token, even when that token is X. The tokens count as generated tokens, toward `max_tokens` (which cuts them short) and `completion_tokens`, and stop ids among them stop the request. The reply goes on from them as from a prompt that ended with them. Not with `logprobs`: an inserted token has none |

Logprobs are of the distribution the token was sampled from **before** temperature/top-p/top-k/min-p are applied
(vLLM's default `logprobs_mode=raw_logprobs`), log-softmax over the full vocab. Rank is 1-based over that same
distribution (rank 1 = argmax).

Speculative decoding is invisible here: the engine may emit several accepted tokens in one `tokens` frame; it
must never emit a token it later retracts. An insert is the same: the engine decides it from the tokens it commits, so
tokens it sampled past the insert are never sent.

### `cancel`

```json
{"type": "cancel", "id": "cmpl-3f2a..."}
```

Stop the request as soon as possible and free its state. The engine answers with a `finish` frame
(`"reason": "abort"`) unless the request already finished. Cancelling an unknown id is a no-op (no reply). The front
end sends it when the HTTP client disconnects, when a text stop string matched, or when a `thinking_budget` closes the
think block. In the last case it waits for the `finish`, then sends a new `generate` (a new id) whose prompt is the old
prompt, the reply so far and the close (frontend/README.md, "Thinking controls").

### `metrics`

```json
{"type": "metrics", "seq": 7}
```

Ask for a counters snapshot; the engine replies with a `metrics` frame carrying the same `seq`.

## Engine -> front end

### `tokens`

```json
{"type": "tokens", "id": "cmpl-3f2a...", "ids": [785, 1196, 6801],
 "logprobs": [{"lp": -0.012, "rank": 1, "top": [[785, -0.012], [1197, -4.9]]},
              {"lp": -0.3,   "rank": 1, "top": [[1196, -0.3],  [220, -1.5]]},
              {"lp": -2.1,   "rank": 3, "top": [[11, -0.4],    [13, -1.3]]}]}
```

- `ids`: one or more newly generated ids (MTP accepts arrive together), in order. Inserted tokens come in the same
  frames as the tokens around them.
- `logprobs`: present only if the request asked for `logprobs`; one entry per id. `lp` and `rank` are for the
  emitted id (`rank` is null when the engine does not know it: the id is below the engine's top 20, or k = 0); `top` is the top-k `[id, logprob]` pairs sorted by logprob descending (k entries; the emitted id may or
  may not be among them).
- The id that triggers a stop (a member of `stop_ids`) **is included** in `ids` and counts in `completion_tokens`
  (vLLM does the same); the front end drops its text.

### `prompt_logprobs`

```json
{"type": "prompt_logprobs", "id": "cmpl-3f2a...",
 "lps": [null, {"lp": -7.2, "rank": 812, "top": [[25062, -0.9], [11, -1.4]]}, ...]}
```

Sent once, before the first `tokens` frame, only if requested. `lps` has exactly `len(prompt)` entries;
entry 0 is `null`. Entry j describes `prompt[j]`, same fields as a `tokens` logprob entry (`rank` may be null, see above;
the `top` list always carries ranks 1..k by its order).
Prefix-cache hits must not drop positions: if the engine cannot produce prompt logprobs for cached positions it
must recompute them (or not use the cache for this request).

### `finish`

```json
{"type": "finish", "id": "cmpl-3f2a...", "reason": "stop", "stop_id": 154827,
 "prompt_tokens": 30, "cached_tokens": 0, "completion_tokens": 212}
```

The last frame for a request; the id may be reused after it.

| field | meaning |
|---|---|
| `reason` | `stop` (a `stop_ids` token was emitted), `length` (hit `max_tokens`), `abort` (cancelled), `error` |
| `stop_id` | the stop token for `reason: stop`, else null |
| `prompt_tokens` | `len(prompt)` |
| `cached_tokens` | prompt tokens served from the prefix cache (0 if none) |
| `completion_tokens` | ids emitted in `tokens` frames for this request |
| `message` | only with `reason: error`: human-readable cause (e.g. `"prompt too long"`, `"KV cache full"`) |

A request that fails validation in the engine gets `finish` with `reason: error` directly (no `tokens`).

### `metrics` (reply)

```json
{"type": "metrics", "seq": 7,
 "spec": {"drafts": 1200, "draft_tokens": 3600, "accepted": 2500, "accepted_per_pos": [1000, 850, 650]},
 "prefix_cache": {"queries": 90000, "hits": 41000},
 "requests": {"running": 1, "waiting": 0, "finished": {"stop": 40, "length": 12, "abort": 1, "error": 0}},
 "tokens": {"prompt": 912345, "generation": 120000}}
```

All values are monotonically increasing counters since engine start, except `requests.running` / `waiting`
(gauges).

- `spec.drafts`: number of verify rounds that had at least one draft token;
  `draft_tokens`: total draft tokens proposed; `accepted`: total draft tokens accepted;
  `accepted_per_pos[i]`: rounds whose draft token at position i (0-based) was accepted. The array length is the
  max draft depth (k); positions the engine never drafts are 0.
- `prefix_cache.queries` / `hits` count **tokens** (prompt tokens looked up / found), like vLLM.
- `tokens.prompt` counts prompt tokens computed (not cached); `tokens.generation` counts generated tokens.

The front end maps these to vLLM's Prometheus names on `/metrics` (see frontend/README.md).

## Error handling summary

- Malformed frame (bad length, bad JSON): the receiver logs and closes the connection (both sides then treat all
  in-flight requests as dead).
- `generate` with a duplicate in-flight id: engine replies `finish` `reason: error`, `message: "duplicate id"`.
- The engine must keep reading frames while generating (a `cancel` must not wait behind a long prefill for more
  than one scheduler step).

## Version history

- v1 (2026-09-30): initial.
- v1 (2026-10-12): `generate` takes an optional `system_tokens` (an engine without KV stores ignores it). The shim
  passes it on as `sys <n>` in the request header.
- v1 (2026-10-12): `generate` `insert` and hello `insert`. An engine without them leaves the hello key out, and the
  front end then sends no `insert`, so the version stays 1.
