# g6pseudo: a pseudo engine

`g6pseudo` speaks [PROTOCOL.md](../../PROTOCOL.md) in GLM-5.3 tokens, as `g6m4` does, and answers from its weights: a
TOML file of rules, [model/responses.toml](model/responses.toml). The front end, its tool-call and reasoning parsers,
streaming, logprobs, `tool_choice`, thinking budgets and the smoketest then run end to end on a CPU, with no GB10 and no
model. It is for CI and for local work on everything above the engine. It tests nothing in the engine itself.

    frontend/pseudo-engine/smoke.sh TOKENIZER_DIR       smoketest/run.sh against the front end and g6pseudo

`TOKENIZER_DIR` holds GLM-5.3's `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and
`generation_config.json`: the release's top level, or a box's model cache `tokenizer/` directory.

    g6pseudo --model DIR [--sock PATH]       serve; DIR holds responses.toml and the tokenizer files (at its top
                                             level or under tokenizer/)
    g6pseudo --model DIR --rank R --serve S2DIR --s2-epoch E   the S2 file protocol, as g6m4 --serve, for the shim
    g6pseudo --weights FILE --check          compile the rules
    g6pseudo --weights FILE --explain TEXT   the rules a user message TEXT matches, their weights, the winner's reply

## Through the launcher

`./gr8 pseudo` boots it as `./gr8 glm53` boots g6m4: the supervisor, the handshake, the shim and the front end, at TP 1
to 4 and with no RDMA. The release is a directory with `responses.toml` and the four tokenizer files, and its model set
id follows the bytes of both. `./gr8 pseudo build` puts `g6pseudo` and `g6front-rs` in `bin/`.

    ./gr8 pseudo build
    ./gr8 pseudo up --peers 127.0.0.1 --model RELEASE_DIR
    ./gr8 pseudo smoke

Under `--serve` (src/s2.rs) rank 0 serves the S2 directory and the other ranks wait for its `quit`. It reports MTP 0
and takes `--cap` as the pool. `tools/launch/test_pseudo_e2e.py` boots it with `GR8_PSEUDO_TOKENIZER` set.

## The weights

A rule is a regex over one part of the prompt and the reply it gives:

```toml
[[rule]]
name = "weather-call"
match = '(?i)weather (?:in|for) (?P<city>[A-Z][\w-]*)'
tools = ["get_weather"]          # the prompt must define these tools ("*": any)
specificity = 10
reasoning = "The user wants the weather in ${city}."
content = "Let me check the weather in ${city}."
[[rule.call]]
name = "get_weather"
arguments = { city = "${city}" }
```

- `on` is the part of the prompt the regex reads: `user` (the last user message, the default), `tool` (the last tool
  result, when the prompt ends with one), `system` (the system messages and the tool block) or `prompt` (all of it).
- Captures go into the reasoning, the content and the arguments as `$1` or `${name}`.
- A reply is the reasoning (`[engine].reasoning` when the rule names none), then the content, then its calls in GLM's
  `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` form. String arguments go in as text,
  other values as JSON.
- A reply with calls ends on `<|observation|>`, and any other reply on `<|user|>`. `fill = true` repeats the content
  until `max_tokens` instead.
- A prompt whose reply already closed its think block (thinking off, or a thinking budget's continuation) gets no
  reasoning. A raw `/v1/completions` prompt, which opens no assistant turn, gets the content alone.
- `[fallback]` answers a prompt no rule matches. `[engine]` sets the hello's model name and `max_model_len`, the pace
  (`tokens_per_s`, `chunk` ids a frame, `prefill_tps`) and the default reasoning.

## Which rule answers

Every rule that matches is a candidate. The most specific wins: the higher `specificity`, then the longer match, then
the rule earlier in the file. To make a new rule beat one that already matches, give it a higher specificity.

A candidate's weight is the softmax of its score: its specificity plus a thousandth of its match length.
- **Logprobs** follow the candidates. At each position, the emitted token's probability is the weight of the candidates
  still in agreement with the reply so far that emit it, over the weight of all those in agreement. So the top
  alternatives are what the next-best rules would have said there.
- **Sampling:** `temperature` 0 takes the winner. A higher temperature draws among the candidates by their weights at
  that temperature, from the request's seed and the prompt, so the same request gets the same reply.
- **`insert` rules** apply as an engine applies them: the reply goes on from the inserted tokens as from a prompt that
  ended with them. A forced `tool_choice` therefore continues inside the call, with the winning rule's call or else
  the prompt's first tool.

Prompt logprobs are made up but well formed, since the rules say nothing about the prompt.

## Adding a rule

Add it to `model/responses.toml`, check it with `--explain`, and run `cargo test` (CI also runs `--check`). Keep rules
deterministic and specific. Regexes cannot do arithmetic, so a rule like the smoketest's `17x23` matches its literal
question.
