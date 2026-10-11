# g6 front end

An OpenAI/vLLM-compatible HTTP server in front of the g6 engine. It talks to the engine's shim over a Unix socket using
the protocol in `../PROTOCOL.md`, and handles tokenization, the chat template, the GLM reasoning and tool parser,
streaming and `/metrics`. It comes in two implementations with the same flags and byte-identical responses
(`tests/diff_fronts.py`): `g6front-rs/`, one static Rust binary, which `./gr8 glm53 up` runs; and `g6front/`, pure
Python asyncio, the reference the tests compare against.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /tokenize`
- `POST /detokenize`
- `GET /v1/models`
- `GET /health`
- `GET /metrics`

## Build and run (on the head)

`./gr8 glm53 up` starts it. By hand, with the shim already listening on `--engine-sock`:

```bash
scripts/build_front.sh                 # -> bin/g6front-rs (rustup fetches the pinned toolchain on first build)
bin/g6front-rs --model-dir <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json> \
  --engine-sock $G6_HOME/run/engine.sock --port 8100
```

The Python implementation instead:

```bash
python3 -m venv $G6_HOME/venv && $G6_HOME/venv/bin/pip install --no-cache-dir -r frontend/requirements.txt
cd frontend && $G6_HOME/venv/bin/python -m g6front --model-dir <dir> --engine-sock $G6_HOME/run/engine.sock --port 8100
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
| `--log-level` | `INFO` | |

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

## Rust front end: g6front-rs

It uses less head-node memory and adds less time to TTFT than the Python one.

Extra flags (the rest are the Python ones):

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
- **Memory goes back to the OS.** It uses mimalloc. Loading `tokenizer.json` peaks at ~280 MB, and the Python process
  keeps most of that resident. Here it is collected after load and after each long encode, off the request path.

Measured against the Python front end in a linux/aarch64 container (12 CPUs, so 6 encode threads;
`tests/bench_fronts.py`, mock engine with instant prefill, each side alone):

| | Python | Rust |
|---|---|---|
| resident memory, idle / after 16 concurrent streams | 211 / 362 MB | 64 / 95 MB |
| TTFT, 7K / 28K / 112K / 225K-token prompt | 8.8 / 31.4 / 151 / 316 ms | 4.7 / 6.5 / 20.7 / 37.7 ms |
| one stream, tokens/s through the front end | 43K | 193K |
| 16 streams, chunks/s / CPU per chunk | 53K / 18.7 µs | 179K / 8.4 µs |

Known differences, outside what the diff harness and the clients send:

- The text of two errors differs: invalid request JSON (`invalid JSON body: ...`) and jinja errors
  (`chat template error: ...`). The status and error type are the same.
- An integer in request JSON that does not fit in 64 bits is read as a float. One example is a huge number inside
  past tool-call arguments in `messages`. The template then renders it in float form. Tool-call arguments parsed from
  model output are exact at any size.
- Some odd request values that make Python raise `TypeError` (HTTP 500) get a 400 here, e.g. `stop: 5`.
- Request JSON is parsed strictly: `NaN`/`Infinity` literals, which Python's `json` accepts, are invalid JSON here.
  So is the same in a JSON-string `arguments` of a past tool call, which then renders as `{}`. A `-0` there renders
  as `-0.0`.
- `top_k` beyond 64 bits is sent to the engine as 0 (off) instead of the exact integer.
- Templates using transformers' `{% generation %}` tag are not supported. GLM's template does not use it.

Tests: `cargo test --release` (unit tests); `tests/diff_fronts.py` (parity with the Python front end over 127 cases,
then the successful ones replayed concurrently); `tests/bench_fronts.py` (the numbers above); and `tests.test_api` and
`bench/pbench.py` run unchanged against it.

## Real engine through files: g6s2shim.py

`g6s2shim.py` is the engine-side shim for `g6m4 --serve DIR`, on the head: it listens on the PROTOCOL.md socket, and up to `--s2-sessions` requests decode in one step. Stdlib only. The engine side of the file protocol is `engine/runtime/s2/{ingress.h,sessions.h,service.h,scheduler.h}`; the shim docstring has the summary:

- The engine writes `DIR/ready.s2` (`G6S2 2 <epoch> <ranks> <pool_rows> <mtp> <guard> <max_rows> <max_lp>`) when every rank has booted. Version 1 (no `<max_lp>`) has no logprobs. The shim sends `hello` only after that, with `max_model_len = pool_rows - guard` (lower it with `--max-model-len`). `DIR/status.s2` (queue, free pages) goes into the `metrics` frame as an extra `s2` field.
- A request is written as `req.<id>` (two lines: `n_gen .. temp .. seed .. min .. stop .. top_p .. [lp <k>] [plp <k>]`, then the prompt ids) to the head's DIR, then `cmd.<k>` = `ADMIT <epoch> <id> <fnv1a64 of the req bytes>`. Ids increase; command numbers have no gap. The head engine sends each request's bytes to the other ranks with its ADMIT, over the control links, so the shim needs no access to them. The engine queues a request FIFO until a session slot and a KV run are free.
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
  [--prefill-tps 50000] [--rounds-per-s 100] [--max-model-len 200000]
tests/serve_mock.sh start|stop      # mock + front end on :8100 (env PY, MODEL_DIR, PORT, ROUNDS, FRONT_ARGS)
python -m unittest tests.test_parser tests.test_api   # with the mock + front end running
```

## Deviations from vLLM

- These return 400: `n>1`, batched prompts, `echo`, `suffix`, named or required `tool_choice`, presence/frequency/repetition penalties, and a custom `chat_template`.
- Tool-call argument JSON streams once per completed key/value pair instead of per token. The final concatenated string is identical.
- Tool-call tags are parsed only when the request has `tools` and `tool_choice` is not `none`.
- `--reasoning-fields both` sends both `reasoning` and `reasoning_content` (the original brief). The harness `common.py` concatenates the two, so `text_head` would show the reasoning twice; the default is therefore `reasoning` (exact vLLM 0.30 parity).
- The benchmark client is `../bench/pbench.py`.
