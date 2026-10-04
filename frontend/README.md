# g6 front end

This is an OpenAI/vLLM-compatible HTTP server that sits in front of the g6 engine. It talks to the engine over a Unix socket using the protocol in `../PROTOCOL.md`. It is pure asyncio and needs no torch or vLLM. It handles tokenization, the chat template, the GLM reasoning/tool parser, streaming and `/metrics`.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /tokenize`
- `POST /detokenize`
- `GET /v1/models`
- `GET /health`
- `GET /metrics`

## Install and run (on the rank-0 node)

```bash
# once: venv (~73 MB, no torch)
python3 -m venv $G6_HOME/venv
$G6_HOME/venv/bin/pip install --no-cache-dir -r frontend/requirements.txt

# run (the shim must already be listening on --engine-sock; scripts/launch_tp6.sh does all of this)
cd frontend && $G6_HOME/venv/bin/python -m g6front \
  --model-dir <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json> \
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

## Real engine through files: g6shim.py

`g6shim.py` listens on the engine socket and drives `g6m4`/`m4sim --serve DIR` through `req.<n>` / `resp.<n>` files (format in its docstring). One request at a time; `--rank-dirs` takes rank 0's local DIR first, then the other ranks as `host:/path` (written over one persistent ssh connection per host). `cancel` answers `abort` at once and swallows the rest of that request. SIGTERM writes `DIR/quit` to every rank. `logprobs` / `prompt_logprobs` (k = 0..`--max-logprobs`, default 20) go to the engine as `lp <k>` / `plp <k>` request keys; the engine answers with `l` / `p` lines (serve.h). `--max-plp-prompt` (default 4096) bounds the prompt length of a `prompt_logprobs` request. The rank of a token beyond the engine's top 20 (and every rank at k = 0) is `null`. top_p is passed; top_k/min_p are ignored with a warning. Tests: `tests/fake_serve_engine.py` (a fake `--serve` engine) and `tests/test_shim.py SOCK`.

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
