# Shared smoketest helpers, sourced by smoketest/run.sh. Kept byte-identical
# across the spark-* container repos.
#
# A case is a shell function named t_<something>; run_all runs them in file
# order. Checks are jq expressions that must be true, so a new kind of check
# never needs a change here -- this file only moves requests and counts results.
#
#     smoketest/run.sh <base-url> [served-name]
#     ONLY=t_thinking_on smoketest/run.sh <base-url>     # one case
#
# Exit status is the number of failed cases.
set -uo pipefail

BASE="${1:?usage: run.sh <base-url> [served-name]}"
BASE="${BASE%/}"
HERE="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
ROOT="$(dirname "$HERE")"
NAME="${2:-$(sed -n 's/^name: *//p' "$ROOT/model.yaml" | head -1)}"
TIMEOUT="${TIMEOUT:-300}"
PASS=0 FAIL=0 CASE="" STATUS="" BODY="" SECS=""
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

for tool in curl jq; do
  command -v "$tool" >/dev/null || { echo "smoketest needs $tool"; exit 2; }
done

# post PATH JSON -> STATUS, BODY, SECS. The body goes through stdin: a single
# argv string is capped at 128 KiB on Linux, and a data URL passes that fast.
post() {
  local out
  out=$(printf '%s' "$2" | curl -s --max-time "$TIMEOUT" -w '\n%{http_code} %{time_total}' \
        -H 'Content-Type: application/json' --data-binary @- "$BASE$1")
  BODY="${out%$'\n'*}"; read -r STATUS SECS <<<"${out##*$'\n'}"
}

# post_form PATH curl-form-args... -> STATUS, BODY, SECS (multipart uploads)
post_form() {
  local path="$1" out; shift
  out=$(curl -s --max-time "$TIMEOUT" -w '\n%{http_code} %{time_total}' "$@" "$BASE$path")
  BODY="${out%$'\n'*}"; read -r STATUS SECS <<<"${out##*$'\n'}"
}

# jqb EXPR: evaluate EXPR against BODY, via a file rather than argv.
jqb() { printf '%s' "$BODY" > "$TMP/body.json"; jq -r "$@" "$TMP/body.json"; }

# check DESCRIPTION JQ-BOOLEAN: fail the case unless EXPR is true of BODY.
check() {
  printf '%s' "$BODY" > "$TMP/body.json"
  if ! jq -e "$2" "$TMP/body.json" >/dev/null 2>&1; then
    FAIL=$((FAIL + 1))
    echo "  FAIL ${CASE#t_}: $1"
    echo "       status $STATUS, body: ${BODY:0:400}"
    return 1
  fi
}

# count_tokens TEXT -> the model's own token count, via /tokenize.
count_tokens() {
  jq -n --arg m "$NAME" --arg p "$1" '{model:$m, prompt:$p, add_special_tokens:false}' \
    | curl -s --max-time 60 -H 'Content-Type: application/json' --data-binary @- "$BASE/tokenize" \
    | jq -r '.count'
}

ok() {
  local s="$SECS"; [ "$s" != - ] && s=$(printf '%.2f' "$s")
  PASS=$((PASS + 1)); echo "  ok   ${CASE#t_}  (${s}s${1:+, $1})"
}

run_all() {
  echo "smoketest $NAME against $BASE"
  local c
  for c in $(grep -oE '^t_[a-z0-9_]+\(\)' "${BASH_SOURCE[1]}" | tr -d '()'); do
    [ -n "${ONLY:-}" ] && [ "$c" != "$ONLY" ] && continue
    CASE="$c"; STATUS=""; BODY=""; SECS="-"
    "$c" || true
  done
  echo "$PASS/$((PASS + FAIL)) passed"
  exit "$FAIL"
}
