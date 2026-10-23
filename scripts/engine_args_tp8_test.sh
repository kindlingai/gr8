#!/bin/bash
# The TP8 arm of scripts/engine_args.sh: its defaults, its refusal (KV_FORMAT=int4), DFLASH=1 allowed, and TP4/TP6 unchanged.
#   scripts/engine_args_tp8_test.sh
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/scripts/engine_args.sh"
fails=0
want() { [ "$2" = "$3" ] || { echo "FAIL $1: got '$2' want '$3'" >&2; fails=$((fails + 1)); }; }
got=$(TP=8; engine_defaults && engine_env && echo "$SHARE_LIMIT $CAP $READY_TMO $MTP $SESS $S2PORT $DFLASH $SPEC_ADAPT $PF_OWN $REUSE $KV_FORMAT $ENGINE_BIN $(( SHARE_LIMIT / 64 + (CAP - SHARE_LIMIT) / (64 * TP) ))")
want "TP8 defaults" "$got" "32768 65536 3600 2 8 7341 0 0 3 1 fp8 g6m4 576"
got=$(TP=8; engine_defaults && engine_env && echo "$ENGINE_ENV" | grep -c 'G6_PF_QP_ROLL')
want "TP8 has no QP roll" "$got" "0"
got=$(TP=8; DFLASH=1; engine_defaults && echo "$DFLASH $SPEC_ADAPT"); want "TP8 DFLASH=1 allowed" "$got" "1 1"
(TP=8; KV_FORMAT=int4; engine_defaults 2> /dev/null); want "TP8 int4 refused" "$?" "2"
(TP=5; engine_defaults 2> /dev/null); want "TP5 refused" "$?" "2"
got=$(TP=4; engine_defaults && echo "$SHARE_LIMIT $CAP $PF_OWN $DFLASH"); want "TP4 unchanged" "$got" "32768 485376 4 1"
got=$(TP=6; engine_defaults && echo "$SHARE_LIMIT $CAP $PF_OWN $DFLASH"); want "TP6 unchanged" "$got" "131072 614400 3 0"
[ "$fails" = 0 ] && echo "engine_args_tp8_test: PASS" || { echo "engine_args_tp8_test: $fails FAIL"; exit 1; }
