#!/bin/bash
# down.sh ENVFILE: stop the TP8 serve on every box and confirm no g6m4 is left (run on the head box as root).
set -u -o pipefail
E=${1:?env file}; H=${H:?gr8 checkout path}; BOXES=${BOXES:?eight ssh addresses, head first}; RC=0
for b in $BOXES; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 root@"$b" "
    for u in \$(systemctl list-units --plain --no-legend 'tp8-*' | awk '{print \$1}'); do systemctl stop \"\$u\"; done
    cd $H && python3 tools/launch/launch.py down --local @$E > /dev/null 2>&1
    python3 tools/launch/launch.py verify-down --kill @$E 2>&1 | tail -1
    n=\$(pgrep -c -x g6m4 || true); echo \"\$(hostname -s) g6m4=\$n\"; [ \"\$n\" = 0 ]" || RC=1
done
exit $RC
