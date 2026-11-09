#!/bin/bash
# boot.sh ENVFILE: bring the five-rank TP5 serve up on every box and leave it up (run on the head box as root).
# Derived from serve-t5x.sh, the script that runs the measured serve; site-only steps (host memory scrub, menu hook) removed.
# BOXES: the five box addresses for ssh, head (rank 0) first. H: the gr8 checkout on every box (same path on all five).
# CTX: the max_model_len to expect (617532 for T5X.env).
set -u -o pipefail
E=${1:?env file}
H=${H:?set H to the gr8 checkout path on the boxes}
BOXES=${BOXES:?set BOXES to the five ssh addresses, head first}
CTX=${CTX:-617532}; U=${U:-tp5-serve}
on(){ local b; for b in $BOXES; do
  ssh -n -o ConnectTimeout=10 -o BatchMode=yes root@"$b" "$1" || { echo "$b: step failed" >&2; return 1; }; done; }
PORT=$(grep -oE '^PORT=[0-9]+' "$E" | cut -d= -f2); PORT=${PORT:-8100}
health(){ [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health)" = 200 ]; }
ctx(){ curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -o '"max_model_len":[0-9]*' | head -1 | cut -d: -f2; }
on "[ -z \"\$(pgrep -x 'g6m4|g6m4-int4')\" ] || { echo g6m4 running; exit 1; }; test -r $E && grep -qE '^TP=5\$' $E && test -x $H/bin/g6m4" || exit 1
# The measured serve ran at a 2000 MHz GPU clock cap with CPU idle states off.
on "nvidia-smi -lgc 0,2000 > /dev/null && cpupower idle-set -D 0 > /dev/null" || exit 1
on "systemctl reset-failed $U 2>/dev/null; mkdir -p $H/out && systemd-run --quiet --collect -p RemainAfterExit=yes -p TimeoutStopSec=30 --setenv=HOME --setenv=PATH --unit=$U /bin/bash -c 'cd $H && exec ./gr8 glm53 up @$E > out/window-up.log 2>&1 < /dev/null'" || exit 1
t0=$(date +%s)
until health; do
  [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "no /health after 900 s" >&2; exit 1; }
  systemctl is-active -q $U || { echo "rank 0 exited: $(tail -3 $H/out/window-up.log)" >&2; exit 1; }
  sleep 2
done
[ "$(ctx)" = "$CTX" ] || { echo "serving, but max_model_len $(ctx) != $CTX" >&2; exit 1; }
echo "TP5 up in $(( $(date +%s) - t0 )) s, max_model_len $CTX"
