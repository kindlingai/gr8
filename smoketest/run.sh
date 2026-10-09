#!/usr/bin/env bash
# Smoketests for g6. See lib.sh for the conventions.
#
# These test what clients rely on, not raw capability. A wrong rank pack, a
# rank order that does not match the packs, or a broken kernel can load,
# report healthy and serve fluent nonsense, so every case asks for an answer
# that can be checked.
#
#     smoketest/run.sh http://<head>:8100 [served-name]      or ./gr8 glm53 smoke
source "$(dirname "$0")/lib.sh" "$@"

get() { # PATH -> STATUS, BODY, SECS
  local out
  out=$(curl -s --max-time "$TIMEOUT" -w '\n%{http_code} %{time_total}' "$BASE$1")
  BODY="${out%$'\n'*}"; read -r STATUS SECS <<<"${out##*$'\n'}"
}

chat() { # JSON-extra MAX_TOKENS USER-TEXT -> request body
  jq -n --arg m "$NAME" --argjson x "$1" --argjson n "$2" --arg q "$3" \
    '{model:$m, max_tokens:$n, temperature:0, messages:[{role:"user", content:$q}]} + $x'
}
OFF='{"chat_template_kwargs":{"enable_thinking":false}}'

# 200 only once the shim answers: the front end starts before the engine is ready.
t_health() {
  get /health
  [ "$STATUS" = 200 ] || { BODY="{\"status\": \"$STATUS\"}"; check "status should be 200" 'false'; return; }
  ok
}

t_models_lists_served_name() {
  get /v1/models
  check "should list $NAME" "[.data[].id] | index(\"$NAME\") != null" || return
  ok "max_model_len $(jqb --arg n "$NAME" '.data[] | select(.id == $n) | .max_model_len')"
}

# A known fact. A rank pack in the wrong place answers fluently and wrongly.
t_fact() {
  post /v1/chat/completions "$(chat "$OFF" 256 'What is the capital of Australia? One word.')"
  check "should answer Canberra" '.choices[0].message.content | test("Canberra")' || return
  ok "$(jqb '.usage.completion_tokens') tok"
}

# Thinking on (the template's default): the answer in the content, the trace
# in the reasoning field.
t_thinking_on_by_default() {
  post /v1/chat/completions "$(chat '{}' 2048 'What is 17*23? Reply with the number only.')"
  check "should finish" '.choices[0].finish_reason == "stop"' || return
  check "content should be 391" '.choices[0].message.content | test("391")' || return
  check "reasoning should be present" '(.choices[0].message.reasoning // .choices[0].message.reasoning_content // "") | length > 0' || return
  ok "$(jqb '.usage.completion_tokens') tok"
}

t_thinking_off_per_request() {
  post /v1/chat/completions "$(chat "$OFF" 256 'What is 17*23? Reply with the number only.')"
  check "content should be 391" '.choices[0].message.content | test("391")' || return
  check "content should hold no think tags" '.choices[0].message.content | test("think>") | not' || return
  ok "$(jqb '.usage.completion_tokens') tok"
}

# Streaming: the deltas assemble into the answer, and the stream ends with [DONE].
t_stream() {
  local body out t0=$SECONDS
  body=$(chat "$OFF" 256 'What is the capital of France? One word.' | jq '. + {stream:true}')
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
  post /v1/chat/completions "$(jq -n --arg m "$NAME" '{
    model:$m, max_tokens:1024, temperature:0, tool_choice:"auto",
    chat_template_kwargs:{enable_thinking:false},
    tools:[{type:"function", function:{name:"get_weather", description:"Get the current weather for a city.",
            parameters:{type:"object", properties:{city:{type:"string"}}, required:["city"]}}}],
    messages:[{role:"user", content:"What is the weather in Paris right now? Use the tool."}]}')"
  check "should call get_weather" '.choices[0].message.tool_calls[0].function.name == "get_weather"' || return
  check "arguments should name Paris" '.choices[0].message.tool_calls[0].function.arguments | fromjson | .city | test("Paris")' || return
  ok
}

# The raw completions endpoint, no chat template.
t_completion() {
  post /v1/completions "$(jq -n --arg m "$NAME" '{model:$m, prompt:"The capital of France is", max_tokens:8, temperature:0}')"
  check "should continue with Paris" '.choices[0].text | test("Paris")' || return
  ok
}

t_logprobs() {
  post /v1/chat/completions "$(chat '{"chat_template_kwargs":{"enable_thinking":false},"logprobs":true,"top_logprobs":3}' 4 'Say hello.')"
  check "should carry one logprob entry per token" '.choices[0].logprobs.content | length == 4' || return
  check "each entry should have 3 alternatives" '[.choices[0].logprobs.content[].top_logprobs | length] | all(. == 3)' || return
  ok
}

t_max_tokens_exact() {
  post /v1/chat/completions "$(chat "$OFF" 5 'Count from one to fifty in words.')"
  check "should stop on length" '.choices[0].finish_reason == "length"' || return
  check "should emit exactly 5 tokens" '.usage.completion_tokens == 5' || return
  ok
}

# The limit comes from the engine (CAP), so this follows a change to it.
t_context_limit_refused() {
  post /tokenize "$(jq -n --arg m "$NAME" '{model:$m, prompt:"x"}')"
  local max; max=$(jqb '.max_model_len')
  post /v1/chat/completions "$(chat "$OFF" "$max" 'Say hi.')"
  [ "$STATUS" = 400 ] || { BODY="{\"status\": $STATUS}"; check "status should be 400" 'false'; return; }
  check "error should name the context length" '(.error.message // .message) | test("context length")' || return
  ok "limit $max"
}

run_all
