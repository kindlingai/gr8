#!/bin/bash
# handboot.sh ENVFILE TAG: bring the six-rank TP6 serve up on every box and leave it up (run on the head box as root).
# Derived from the script that booted the measured serve; site-only steps (host memory scrub, canary, diagnostics) removed.
# BOXES: the six box addresses for ssh, head first. H: the gr8 checkout on every box (same path on all six).
set -u -o pipefail
E=${1:?env file}; TAG=${2:?tag}
H=${H:?set H to the gr8 checkout path on the boxes}
BOXES=${BOXES:?set BOXES to the six ssh addresses, head first}
RD=${RD:-/tmp/tp6-$TAG-$(date -u +%m%dT%H%M%SZ)}; mkdir -p "$RD" || exit 2
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$RD/log"; }
fan(){ local tag=$1 cmd=$2 b bad=0 pids=""
  for b in $BOXES; do ( ssh -n -o BatchMode=yes -o ConnectTimeout=10 root@"$b" "$cmd" > "$RD/$tag.$b" 2>&1; echo $? > "$RD/rc.$tag.$b" ) & pids="$pids $!"; done
  wait $pids
  for b in $BOXES; do [ "$(cat "$RD/rc.$tag.$b")" = 0 ] || { say "  $tag FAILED on $b: $(tail -2 "$RD/$tag.$b" | tr '\n' ' ')"; bad=1; }; done
  return $bad; }
say "boot $TAG env $E"
fan pre "[ -z \"\$(pgrep -x 'g6m4|g6m4-int4')\" ] || { echo g6m4 running; exit 1; }; test -r $E && grep -qE '^TP=6$' $E" || exit 1
# The measured serve ran at a 2000 MHz GPU clock cap with CPU idle states off.
fan prep "rm -f /dev/shm/psm_* /dev/shm/g6m4* > /dev/null 2>&1; nvidia-smi -lgc 0,2000 > /dev/null && cpupower idle-set -D 0 > /dev/null" || exit 1
U=tp6-$TAG-$(date -u +%H%M%S); T0=$(date +%s)
fan up "mkdir -p $H/out && systemd-run --quiet --collect -p RemainAfterExit=yes -p TimeoutStopSec=30 --setenv=HOME --setenv=PATH --unit=$U /bin/bash -c 'cd $H && exec ./gr8 glm53 up @$E > out/window-up.log 2>&1 < /dev/null'" || exit 1
ok=0; PORT=$(grep -oE '^PORT=[0-9]+' "$E" | cut -d= -f2); PORT=${PORT:-8100}
for i in $(seq 600); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health)" = 200 ] && { ok=1; break; }
  systemctl is-active -q "$U" || { say "unit $U died after $(( $(date +%s) - T0 ))s"; break; }
  sleep 1; done
say "health ok=$ok after $(( $(date +%s) - T0 ))s unit=$U log=$H/out/window-up.log"
[ $ok = 1 ]
