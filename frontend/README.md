# gr8 front end

An OpenAI/vLLM-compatible HTTP server in front of the gr8 engine. It talks to the engine's shim over a Unix socket using
the protocol in `../PROTOCOL.md`, and handles tokenization, the chat template, the GLM reasoning and tool parser,
streaming and `/metrics`. It is one static Rust binary, `g6front-rs/`, which `./gr8 glm53 up` runs. The Python code
under `frontend/` is the shim and the mock engine (and the frame codec they share, `wire.py`), not a second front end.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /tokenize`
- `POST /detokenize`
- `GET /v1/models`
- `GET /health`; `GET /health?deep` (localhost only): the cluster's self-checks, below
- `GET /metrics`

## /health?deep

From the head itself, `curl localhost:8100/health?deep` asks the head's supervisor (`--supervisor HOST:PORT`, set by
`./gr8 glm53 up`) to check every rank, and answers with one JSON object: 200 when every rank passes, else 503. Other
addresses get 403. Each rank's engine sums its weights on the GPU, on a low-priority stream while it serves, and compares
them with the sums it took once it was ready to serve (engine/runtime/weight_check.cuh): it reads every weight byte once,
about 0.3 s at TP=4. Plain `/health` is unchanged.

```json
{"t": "check", "ok": true, "time": 1760371200.5,
 "ranks": [{"rank": 0, "ok": true, "state": "serving", "mem_gib": 7.0, "compaction_proactiveness": 0,
            "weights": {"rank": 0, "ok": true, "ms": 310, "images": [{"what": "exlr8 release", "bytes": 78490000000,
                        "chunks": 2340, "changed": 0, "digest": "9c0e..."}]}}, ...]}
```

A rank whose weights changed has `"changed": N` and, for the first chunk that differs, `"first": {"offset": ..., "end":
..., "starts_in": "..."}`: its bytes, and the tensor it starts in (a 32 MB chunk can hold several tensors).

## Build and run (on the head)

`./gr8 glm53 up` starts it. By hand, with the shim already listening on `--engine-sock`:

```bash
scripts/build_front.sh                 # -> bin/g6front-rs (rustup fetches the pinned toolchain on first build)
bin/g6front-rs --model-dir <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json> \
  --engine-sock $G6_HOME/run/engine.sock --port 8100
```

On a GB10 (Grace) rank-0 node, tokenizer encode takes 12 ms at 8K tokens and 71 ms at 32K tokens.

## Flags

| flag | default | meaning |
|---|---|---|
| `--model-dir` | required | Directory holding `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json`. |
| `--engine-sock` | `/run/g6/engine.sock` | Engine Unix socket. |
| `--host` / `--port` | `0.0.0.0` / `8100` | Listen address. |
| `--served-model-name` | `glm-5.3 glm53` | Accepted `model` values. The first one is reported in responses. |
| `--max-model-len` | 0 (use the engine `hello`) | Overrides the context length. |
| `--reasoning-fields` | `reasoning` | `reasoning`, `reasoning_content` or both. The default matches vLLM 0.30, which sends only `reasoning`. |
| `--enable-prompt-tokens-details` | off | Adds `usage.prompt_tokens_details.cached_tokens`. |
| `--default-max-tokens` | 0 | `max_tokens` for a chat request that names none, capped by the room the context leaves. 0 = see `--share-tokens`. `./gr8 glm53 up` passes `DEFAULT_MAX_TOKENS` when it is set. |
| `--share-tokens` | 0 | With `--default-max-tokens 0`: a chat request that names no `max_tokens` gets what keeps prompt + `max_tokens` within N, but at least 8,192 (past the limit it runs alone anyway). 0 = all the room the context leaves. `./gr8 glm53 up` passes `SHARE_LIMIT` less a 64-token page for the engine's guard rows. |
| `--max-connections` | 512 | Connections served at once; more wait in the listen backlog. Request headers must arrive within 30 s and a body within 120 s. |
| `--default-reasoning-effort` | none | `reasoning_effort` for a chat request that names none: `minimal`, `low`, `medium`, `high`, `xhigh` or `max`, heard as the nearest level the template names. `./gr8 glm53 up` passes `DEFAULT_REASONING_EFFORT` when it is set. |
| `--default-thinking-budget` | 0 | `thinking_budget` for a chat request that names none, 0 or null (0 = no limit). A request with `logprobs` gets none. `./gr8 glm53 up` passes `DEFAULT_THINKING_BUDGET` when it is set. |
| `--log-level` | `INFO` | |

## Thinking controls (Rust front end)

Read as TensorFold 0.6.1 reads them (`server/request_options.py`). The code is `g6front-rs/src/thinking.rs` and
`gen.rs`.

| field | accepted | effect |
|---|---|---|
| `reasoning_effort`, top level or `chat_template_kwargs.reasoning_effort` (top level wins) | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; anything else is a 400 | The template gets the nearest level it names, on the ladder max > xhigh > high > medium > low > minimal, ties going higher. The names come from the template's text (a quoted `minimal`, `low`, `medium`, `high`, `xhigh` or `max`): GLM 5.3's are low, high and max. A template that names none gets the value as sent. `none` is a 400, unless `enable_thinking` is `true`, which wins and leaves the effort unset (as in TensorFold) |
| `chat_template_kwargs.enable_thinking` | `true`, or unset | `false` (or another false value) is a 400: GLM 5.3's template cannot turn thinking off. The front end adds no `enable_thinking` kwarg of its own |
| `thinking_budget` (top level) | an integer, an integral float, a string holding an integer, or null. Anything else is a 400 | 0, null or unset: `--default-thinking-budget` (TensorFold's `int(budget or default)`). Negative: no limit. N: while the think block is open, the reply's Nth token is replaced by `\n</think>\n\n`, even when it is `</think>` itself, so N-1 reasoning tokens come first (TensorFold's ThinkBudget). `max_tokens` may cut the close short. It applies to a chat prompt that leaves a think block open. The close has no logprobs, so naming both `thinking_budget` and `logprobs` is a 400 |
| `response_format` (chat and completions) | absent, null, `{"type": "text"}` | `json_object`, `json_schema` and any other type are a 400 until the engine has constrained decoding |

`/tokenize` with `messages` applies the same effort and refusals, so its count matches the chat request's prompt.
Chat usage carries `completion_tokens_details.reasoning_tokens`: the reply's tokens through its `</think>`, or all of
them when the reply never closes its think block (TensorFold counts the same way). The response and the last stream
chunk carry the count. The `continuous_usage_stats` chunks leave the count out.

How a budget cuts. When the engine's hello says `"insert": true` (g6m4 since ready.s2 version 3, and the mock), the
front end sends the budget as `generate`'s insert rule (PROTOCOL.md): after N - 1 reply tokens without `</think>`, the
close takes the next token's place. The engine applies it at the token itself, on every rank, and the reply goes on in
the same run (engine/runtime/s2_batch.h). The close's three tokens cost one more main step and its draft.

With an engine that lacks it, the cut ends the engine run (a `cancel`, then its `finish` is awaited) and the reply goes
on in a new `generate` whose prompt is the prompt, the reply so far and the close, with `max_tokens` and `min_tokens`
less the reply's length. The client sees one response or one SSE stream either way. Usage counts every reply token
once: the close counts, and tokens the cut run made past the cut do not. `cached_tokens` is the first run's. The
continuation is a second request to the engine, and S2's prefix reuse keeps only the prompt's whole 4,096-row prefill
cells (decoded rows never count), so it prefills the prompt's last partial cell, the reasoning and the close again. On
the mock with S2's reuse rule and TP=4's cold prefill rate (`--reuse-block 4096 --prefill-tps 1374 --no-insert`):

| prompt | budget | rows prefilled again | added time |
|---|---|---|---|
| 1.1K | 5 / 512 / 2,048 | 1,111 / 1,604 / 3,140 (the whole prompt: under one cell) | 0.8 / 1.2 / 2.3 s |
| 7.5K | 5 / 512 / 2,048 | 3,486 / 4,002 / 5,547 | 2.6 / 2.9 / 4.1 s |
| 32K | 5 / 512 / 2,048 | 3,566 / 4,129 / 5,721 | 2.5 / 3.0 / 4.2 s |

The front end logs each continuation's cached tokens (`thinking budget: the continuation took N of its M prompt
tokens from the cache`).

## Metrics mapping

Engine `metrics` frame → Prometheus names (the same names vLLM uses, so the existing scrapers work):

| engine field | metric |
|---|---|
| `requests.running` / `requests.waiting` | `vllm:num_requests_running` / `vllm:num_requests_waiting` |
| `prefix_cache.queries` / `.hits` | `vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total` |
| `tokens.prompt` / `.generation` | `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` |
| `requests.finished{stop,length,abort,error}` | `vllm:request_success_total{finished_reason=...}` |
| `spec.drafts` / `.draft_tokens` / `.accepted` | `vllm:spec_decode_num_drafts_total` / `_num_draft_tokens_total` / `_num_accepted_tokens_total` |
| `spec.accepted_per_pos[i]` | `vllm:spec_decode_num_accepted_tokens_per_pos_total{position="i"}` |

## Inside it

The notes below are why it is one static Rust binary; the numbers were measured against the Python implementation it
replaced, which has since been removed.

Flags beyond those above:

| flag | default | meaning |
|---|---|---|
| `--workers` | `2` | HTTP/engine I/O threads. About 8 µs of CPU per streamed chunk, so two are plenty; the head also runs rank 0. |
| `--encode-threads` | half the allowed CPUs, at most 8 | Threads for encoding long prompts in parallel. Both pools size from the CPU mask that `taskset` sets in the launch scripts. |

How it differs inside (not in its output):

- **Engine frames are routed without copies.** Each frame is a refcounted slice of the socket read buffer. The reader
  only parses `type`/`id` (borrowed) and hands the slice to its request, which parses the rest in place.
  `prompt_logprobs` is transcoded from the raw frame straight into the response, one position at a time.
- **JSON is written directly.** Responses, SSE events and the `generate` frame are written byte by byte into one
  buffer each, with CPython's float `repr` and escaping. There is no intermediate tree. SSE events go to hyper with
  backpressure.
- **Long prompts are encoded in parallel.** The text is cut only where GLM's pre-tokenizer regex always ends a token:
  after `\n` + ASCII letter, or before a space that follows an ASCII letter. Concatenated ids are identical to one
  encode; `src/tok.rs` gives the argument. A start-up self-check and a fuzz test (`G6_TOKENIZER_DIR=<dir> cargo test
  --release`) confirm it. The self-check turns the parallel path off for a tokenizer whose shape the argument does not
  cover.
- **Memory goes back to the OS.** It uses mimalloc. Loading `tokenizer.json` peaks at ~280 MB, and it is collected
  after load and after each long encode, off the request path, instead of staying resident.

Measured in a linux/aarch64 container (12 CPUs, so 6 encode threads; mock engine with instant prefill), beside the
Python implementation it replaced:

| | Python | this one |
|---|---|---|
| resident memory, idle / after 16 concurrent streams | 211 / 362 MB | 64 / 95 MB |
| TTFT, 7K / 28K / 112K / 225K-token prompt | 8.8 / 31.4 / 151 / 316 ms | 4.7 / 6.5 / 20.7 / 37.7 ms |
| one stream, tokens/s through the front end | 43K | 193K |
| 16 streams, chunks/s / CPU per chunk | 53K / 18.7 µs | 179K / 8.4 µs |

Known rough edges, outside what the clients send:

- The text of two errors differs: invalid request JSON (`invalid JSON body: ...`) and jinja errors
  (`chat template error: ...`). The status and error type are the same.
- An integer in request JSON that does not fit in 64 bits is read as a float. One example is a huge number inside
  past tool-call arguments in `messages`. The template then renders it in float form. Tool-call arguments parsed from
  model output are exact at any size.
- Some odd request values that a Python `json` stack raises `TypeError` on (HTTP 500) get a 400 here, e.g. `stop: 5`.
- Request JSON is parsed strictly: `NaN`/`Infinity` literals, which CPython's `json` accepts, are invalid JSON here.
  So is the same in a JSON-string `arguments` of a past tool call, which then renders as `{}`. A `-0` there renders
  as `-0.0`.
- `top_k` beyond 64 bits is sent to the engine as 0 (off) instead of the exact integer.
- Templates using transformers' `{% generation %}` tag are not supported. GLM's template does not use it.

Tests: `cargo test --release` (unit tests); `python -m unittest tests.test_api` with `tests/serve_mock.sh` up (HTTP,
streaming, tools, thinking); and `bench/pbench.py` runs against it unchanged.

## Real engine through files: g6s2shim.py

`g6s2shim.py` is the engine-side shim for `g6m4 --serve DIR`, on the head: it listens on the PROTOCOL.md socket, and up to `--s2-sessions` requests decode in one step. Stdlib only. The engine side of the file protocol is `engine/runtime/s2/{ingress.h,sessions.h,service.h,scheduler.h}`; the shim docstring has the summary:

- The engine writes `DIR/ready.s2` (`G6S2 3 <epoch> <ranks> <pool_rows> <mtp> <guard> <max_rows> <max_lp>`) when every rank has booted. Version 1 (no `<max_lp>`) has no logprobs, version 2 no insert rule: the shim's hello says `"insert": true` from version 3, and refuses `insert` before it. The shim sends `hello` only after that, with `max_model_len = pool_rows - guard` (lower it with `--max-model-len`). `DIR/status.s2` (queue, free pages, the pages sessions hold and could grow to) goes into the `metrics` frame as an extra `s2` field.
- A request is written as `req.<id>` (two lines: `n_gen .. temp .. seed .. min .. stop .. top_p .. [lp <k>] [plp <k>] [ins <after>:<unless or -1>:<ids>]`, then the prompt ids) to the head's DIR, then `cmd.<k>` = `ADMIT <epoch> <id> <fnv1a64 of the req bytes>`. Ids increase; command numbers have no gap. The head engine sends each request's bytes to the other ranks with its ADMIT, over the control links, so the shim needs no access to them. The engine queues a request FIFO until a session slot and a KV run are free.
- Rank 0 answers in `resp.<id>`: `a` (admitted), `p` (prompt logprobs, before the first `t`), `l` (one per token of the next `t` line, requests with `lp`; both in `engine/runtime/serve.h`), `t <ids>` (tokens frame), `m <rounds> <drafts> <accepted> <per-position>` (spec counters, summed into `metrics`), `end <reason> <stop_id> <n>` (finish frame).
- `cancel` (and a closed front-end connection) is a real `CANCEL <epoch> <id> 0`: a queued request ends `abort` with 0 tokens, a resident one frees its slot and KV run at once. The other sessions do not notice.
- The engine ends the whole epoch on a request it cannot parse or fit, so the shim checks prompt, ids (0 to 154879: the engine does not bound-check the embedding index), `max_tokens`, `top_p` (must be in (0, 1]; 0 is an error, not off), and the length first and answers `error` itself. `logprobs` / `prompt_logprobs` (k = 0..min(`--max-logprobs`, the ready line's `<max_lp>`)) go to the engine as `lp <k>` / `plp <k>`. A `prompt_logprobs` request runs its whole prompt as 8-row decode steps, and every other session waits until that is done: an 813-token prompt held a decoding session for 10.9 s (TP4, about 13.4 ms per prompt token). `--max-plp-prompt` defaults to 1,024 tokens here (about 14 s). `top_k`/`min_p` are ignored with a warning (`top_p` and `min_tokens` go to the engine).
- `DIR/abort.s2` (epoch lost) turns every open request into `error` and every later request too; restart the engine and the shim. SIGTERM writes `DIR/quit`, and the head engine passes it to the other ranks. A restarted shim continues the `req`/`cmd` numbers from the files in DIR. Every rank's DIR must be fresh for a new engine start.

```bash
# engine (every rank; fresh DIR each start): --graph --mtp --prefill, G6_DEC_SPLIT=3, a G6_CP build
g6m4 ... --serve /run/g6/s2 --s2-port N --s2-sessions 4 --s2-epoch 7
python3 g6s2shim.py --dir /run/g6/s2 --sock /run/g6/engine.sock      # on the head only
bin/g6front-rs --engine-sock /run/g6/engine.sock ...
```

Tests, no GPU (`tests/fake_s2_engine.py` is the file side of `g6m4 --serve`: FIFO admit up to `--sessions`, one round time for any number of sessions, per-request `resp`, the engine's strict parsing; tokens depend on prompt and position only, so streams are checked for cross-talk):

```bash
cd frontend && python3 -m unittest tests.test_s2shim -v
```

It covers 4 concurrent streams (exact tokens, wall clock under 2x one stream, 4 resident), resident and queued cancel, a dropped connection, validation errors that must not kill the engine, stop/`min_tokens`, metrics, two rank dirs, a duplicate id, `abort.s2`, SIGTERM, and a shim restart.

### Round log and cost fit

`G6_S2_ROUNDLOG=1 g6m4 ... --serve ...` makes rank 0 print one line per decode step on stderr: `# s2_round step N <main|draft|chain> round R active A S <sessions in this step> M <rows> ms <launch+fence> path <solo|reuse|capture>` (a round with more sessions than fit one step of `max_rows` runs in groups: `S` is the group's share, `A` the whole round's session count, `R` the `Batch::round` id; -1 / 0 outside a round), and on main steps `ex_mean`, `ex_sum`, `ex <per-layer distinct experts>` (from the route log; the main model's route copies are allocated only with this flag). Off, it costs nothing. Fit it:

```bash
tools/s2_costfit.py engine_rank0.log [--residuals] [--json]   # also ms per whole round by active sessions; step_ms = base + per_expert * experts(d*rows) + per_token * rows
tools/s2_costfit.py --selftest
```

## Mock engine and tests

`mock_engine.py` speaks PROTOCOL.md with no model, so the front end and the old benchmark tools can run anywhere:

```bash
python mock_engine.py --model-dir <tokenizer dir> --sock /tmp/g6-mock.sock \
  [--prefill-tps 50000] [--rounds-per-s 100] [--max-model-len 200000] [--reuse-block 64] [--reasoning-repeat 1]
tests/serve_mock.sh start|stop      # mock + front end on :8100 (env PY, MODEL_DIR, PORT, ROUNDS, FRONT, FRONT_ARGS, MOCK_ARGS)
python -m unittest tests.test_api                     # with the mock + front end running
```

`--reuse-block 4096` reuses prefixes as S2 does, in whole 4,096-row prefill cells. `--reasoning-repeat N` makes the
canned reasoning N times longer, for thinking budgets past its 17 tokens. `serve_mock.sh` runs `../bin/g6front-rs`
(`FRONT` names another binary), which is the front end.

## Deviations from vLLM

- These return 400: `n>1`, batched prompts, `echo`, `suffix`, presence/frequency/repetition penalties, a custom `chat_template`, `response_format` other than text, and a request to turn thinking off (above).
- A tool call's arguments arrive when the call is confirmed, not as they are written: nothing in the text says whether
  the model is making a call or writing about one, so the parser waits for the reply to end with it
  (`g6front-rs/src/parser.rs`). The final concatenated string is identical.
- Tool-call tags are parsed only when the request has `tools` and `tool_choice` is not `none`.
- `tool_choice: "required"` and `tool_choice: {"type": "function", "function": {"name": ...}}` force the reply to be a call: the engine's insert rule puts the thinking close and the tool-call opener in place of the reply's first token, and the name too for a named choice, so the model cannot answer in text. The close leads because the template leaves `<think>` open at the end of a chat prompt, so without it the forced call would be reasoning text and the parser would read the reply as prose. It is a strong nudge rather than a grammar (the engine samples, and the front end cannot mask logits): a call to a name the request did not offer comes back as text, and the arguments are the model's own, coerced by the tool's schema. A forced reply carries no reasoning (the inserted close ends the block before any reasoning is written), its inserted tokens count toward `completion_tokens`, and it needs an engine whose hello says `insert` — the same capability as `thinking_budget`. It is refused with `logprobs` (an inserted token has no logprob) and with an explicit `thinking_budget` (one insert rule per request).
- `--reasoning-fields both` sends both `reasoning` and `reasoning_content` (the original brief). The harness `common.py` concatenates the two, so `text_head` would show the reasoning twice; the default is therefore `reasoning` (exact vLLM 0.30 parity).
- The benchmark client is `../bench/pbench.py`.
