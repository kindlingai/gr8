#!/usr/bin/env bash
# Smoketests for DeepSeek-V4.1-Flash-1017 on gr8 (the ds41 front-end dialect). See lib.sh for the conventions.
#
# These test what clients rely on: the served name, a one-prompt "42" check, the thinking switch,
# DSML tool calls and a forced tool_choice. A wrong rank pack or a broken kernel can load, report healthy and serve fluent
# nonsense, so every case asks for an answer that can be checked.
#
#     smoketest/ds41.sh http://<head>:8100 [served-name]      or ./gr8 ds41 smoke
NAME_DEFAULT=deepseek-v4.1-flash
set -- "${1:-}" "${2:-$NAME_DEFAULT}"
source "$(dirname "$0")/lib.sh" "$@"

# DeepSeek-V4.1-Flash-1017's tokenizer: </think> is one token.
THINK_CLOSE=128822

get() { # PATH -> STATUS, BODY, SECS
  local out
  out=$(curl -s --max-time "$TIMEOUT" -w '\n%{http_code} %{time_total}' "$BASE$1")
  BODY="${out%$'\n'*}"; read -r STATUS SECS <<<"${out##*$'\n'}"
}

chat() { # JSON-extra MAX_TOKENS USER-TEXT -> request body
  jq -n --arg m "$NAME" --argjson x "$1" --argjson n "$2" --arg q "$3" \
    '{model:$m, max_tokens:$n, temperature:0, messages:[{role:"user", content:$q}]} + $x'
}

status_is() { # CODE: fail the case unless the last request answered CODE
  [ "$STATUS" = "$1" ] || { BODY=$(jq -n --arg s "$STATUS" --arg b "${BODY:0:300}" '{status:$s, body:$b}'); check "status should be $1" 'false'; return 1; }
}

WEATHER='[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.",
          "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

# 200 only once the shim answers: the front end starts before the engine is ready.
t_health() {
  get /health
  status_is 200 || return
  ok
}

t_models_lists_served_name() {
  get /v1/models
  check "should list $NAME" "[.data[].id] | index(\"$NAME\") != null" || return
  ok "max_model_len $(jqb --arg n "$NAME" '.data[] | select(.id == $n) | .max_model_len')"
}

# One prompt, thinking off, the answer 42.
t_smoke_42() {
  post /v1/chat/completions "$(chat '{"chat_template_kwargs":{"thinking":false}}' 32 'What is 19 + 23? Reply only with the number.')"
  status_is 200 || return
  check "content should be 42" '(.choices[0].message.content // "") | gsub("^\\s+|\\s+$"; "") == "42"' || return
  ok "$(jqb '.usage.completion_tokens') tok"
}

# Served sampling is per request and the engine clears its caches between requests (no prefix reuse unless REUSE=1). Every
# request here carries its temperature (an omitted one is 1.0 for this checkpoint, not 0.7). Greedy, then T 0.7 with top_p 0.95 (the
# nucleus path), then greedy again: the sampled reply still holds the answer and the second greedy reply equals the first.
t_greedy_sampled_greedy() {
  local q='What is 19 + 23? Reply only with the number.' k='{"chat_template_kwargs":{"thinking":false}}' a b c
  post /v1/chat/completions "$(chat "$k" 32 "$q")"
  status_is 200 || return
  a=$(jqb '.choices[0].message.content // ""')
  post /v1/chat/completions "$(chat "$k" 32 "$q" | jq '. + {temperature:0.7, top_p:0.95}')"
  status_is 200 || return
  check "the T 0.7 reply should hold 42" '(.choices[0].message.content // "") | test("42")' || return
  b=$(jqb '.choices[0].message.content // ""')
  post /v1/chat/completions "$(chat "$k" 32 "$q")"
  status_is 200 || return
  c=$(jqb '.choices[0].message.content // ""')
  BODY=$(jq -n --arg a "$a" --arg c "$c" '{first:$a, third:$c}')
  check "the second greedy reply should equal the first" '.first == .third' || return
  ok "T 0.7 said: ${b:0:24}"
}

# Thinking on (the V4.1 default): the trace in the reasoning field, the answer in the content, and the close of the think
# block is the one token 128822 in the reply's ids.
t_thinking_on() {
  post /v1/chat/completions "$(chat '{"chat_template_kwargs":{"thinking":true},"return_token_ids":true}' 4096 'What is 17*23? Reply with the number only.')"
  status_is 200 || return
  check "should finish" '.choices[0].finish_reason == "stop"' || return
  check "content should be 391" '.choices[0].message.content | test("391")' || return
  check "reasoning should be present" '(.choices[0].message.reasoning // .choices[0].message.reasoning_content // "") | length > 0' || return
  check "the reply's ids should hold </think> ($THINK_CLOSE) once" "[.choices[0].token_ids[] | select(. == $THINK_CLOSE)] | length == 1" || return
  ok "$(jqb '.usage.completion_tokens') tok"
}

# thinking off, three ways: the reply has content and no reasoning.
t_thinking_off() {
  local x
  for x in '{"chat_template_kwargs":{"thinking":false}}' '{"chat_template_kwargs":{"enable_thinking":false}}' '{"reasoning_effort":"none"}'; do
    post /v1/chat/completions "$(chat "$x" 256 'Say hi.')"
    status_is 200 || return
    check "content should be present" '(.choices[0].message.content // "") | length > 0' || return
    check "reasoning should be absent" '(.choices[0].message.reasoning // .choices[0].message.reasoning_content // "") == ""' || return
  done
  ok "3 spellings"
}

# reasoning_effort is a whole number 1 to 100 (or a name): past the range is a 400 that says so.
t_reasoning_effort_range() {
  post /v1/chat/completions "$(chat '{"reasoning_effort":101}' 16 'Say hi.')"
  status_is 400 || return
  check "error should name reasoning_effort" '(.error.message // .message) | test("reasoning_effort")' || return
  ok
}

# Streaming: the deltas assemble into the answer, and the stream ends with [DONE].
t_stream() {
  local body out t0=$SECONDS
  body=$(chat '{"chat_template_kwargs":{"thinking":false}}' 64 'What is the capital of France? One word.' | jq '. + {stream:true}')
  out=$(printf '%s' "$body" | curl -sN --max-time "$TIMEOUT" -H 'Content-Type: application/json' \
        --data-binary @- "$BASE/v1/chat/completions")
  STATUS=200; SECS=$(( SECONDS - t0 ))
  BODY=$(printf '%s\n' "$out" | sed -n 's/^data: //p' | jq -sR '
    split("\n") | map(select(length > 0)) as $l
    | {done: ($l | any(. == "[DONE]")),
       content: ([$l[] | select(. != "[DONE]") | fromjson | .choices[0].delta.content // empty] | join(""))}')
  check "stream should end with [DONE]" '.done' || return
  check "content should be Paris" '.content | test("Paris")' || return
  ok
}

t_tool_call_parsed() {
  post /v1/chat/completions "$(jq -n --arg m "$NAME" --argjson t "$WEATHER" '{
    model:$m, max_tokens:1024, temperature:0, tool_choice:"auto", tools:$t,
    chat_template_kwargs:{thinking:false},
    messages:[{role:"user", content:"What is the weather in Paris right now? Use the tool."}]}')"
  status_is 200 || return
  check "should call get_weather" '.choices[0].message.tool_calls[0].function.name == "get_weather"' || return
  check "arguments should name Paris" '.choices[0].message.tool_calls[0].function.arguments | fromjson | .city | test("Paris")' || return
  check "finish reason should be tool_calls" '.choices[0].finish_reason == "tool_calls"' || return
  ok
}

# A forced choice is a call even when the message asks for prose: the front end puts the think close and the call opener
# (named: the invoke line too) in place of the reply's first tokens, so the model cannot answer in text, nor reason first.
t_tool_choice_forced() {
  local tc
  for tc in '"required"' '{"type":"function","function":{"name":"get_weather"}}'; do
    post /v1/chat/completions "$(jq -n --arg m "$NAME" --argjson tc "$tc" --argjson t "$WEATHER" '{
      model:$m, max_tokens:256, temperature:0, tool_choice:$tc, tools:$t,
      messages:[{role:"user", content:"Say hello in French."}]}')"
    status_is 200 || return
    check "should call get_weather anyway" '.choices[0].message.tool_calls[0].function.name == "get_weather"' || return
    check "finish reason should be tool_calls" '.choices[0].finish_reason == "tool_calls"' || return
    check "a forced reply has no reasoning" '(.choices[0].message.reasoning // .choices[0].message.reasoning_content // "") == ""' || return
    ok "$tc"
  done
}

run_all
