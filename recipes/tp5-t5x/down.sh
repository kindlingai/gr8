#!/bin/bash
# down.sh ENVFILE: stop the TP5 serve on every box and confirm no g6m4 is left (run on the head box as root).
set -u -o pipefail
E=${1:?env file}; H=${H:?gr8 checkout path}; BOXES=${BOXES:?five ssh addresses, head first}; U=${U:-tp5-serve}; RC=0
( cd "$H" && timeout 180 ./gr8 glm53 down @"$E" ) || echo "gr8 down rc=$? (the per-box sweep below decides)"
for b in $BOXES; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 root@"$b" "
    for i in \$(seq 45); do pgrep -x 'g6m4|g6m4-int4' > /dev/null || break; sleep 2; done
    systemctl stop $U 2> /dev/null
    n=\$(pgrep -c -x 'g6m4|g6m4-int4' || true); echo \"\$(hostname -s) g6m4=\$n\"; [ \"\$n\" = 0 ]" || RC=1
done
exit $RC
