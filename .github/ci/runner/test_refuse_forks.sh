#!/bin/bash
# refuse-forks.sh against event payloads: a fork's pull request (or a review of one) is refused, ours and pushes are not.
set -u
hook=$(dirname "$0")/refuse-forks.sh
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
fails=0
case_() {   # case_ NAME WANT(run|refuse) EVENT_JSON
  printf '%s' "$3" > "$t/event.json"
  if GITHUB_EVENT_PATH="$t/event.json" GITHUB_REPOSITORY=kindlingai/gr8 "$hook" 2> /dev/null; then got=run; else got=refuse; fi
  [ "$got" = "$2" ] || { echo "FAIL $1: $got, want $2"; fails=$((fails + 1)); }
}
case_ 'fork pull request' refuse '{"pull_request":{"head":{"repo":{"full_name":"someone/gr8"}}}}'
case_ 'deleted fork' refuse '{"pull_request":{"head":{"repo":null}}}'
case_ 'review of a fork pull request' refuse '{"review":{},"pull_request":{"head":{"repo":{"full_name":"someone/gr8"}}}}'
case_ 'pull request from a branch here' run '{"pull_request":{"head":{"repo":{"full_name":"kindlingai/gr8"}}}}'
case_ 'push' run '{"ref":"refs/heads/main","repository":{"full_name":"kindlingai/gr8"}}'
[ $fails = 0 ] && echo "test_refuse_forks: PASS" || exit 1
