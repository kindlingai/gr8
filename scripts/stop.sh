#!/bin/bash
# Stop the front end, the shim and every engine rank (six or four) started by launch_tp6.sh / launch_tp4.sh / launch_s2.sh.
set -u
. "$(dirname "$0")/common.sh"
for p in front shim; do [ -f "$G6_HOME/out/$p.pid" ] && kill "$(cat "$G6_HOME/out/$p.pid")" 2>/dev/null; done
for ((r = 0; r < NRANK; r++)); do on $r "touch $G6_HOME/s2/quit 2>/dev/null" & done; wait   # each rank leaves its S2 loop
sleep 10
for ((r = 0; r < NRANK; r++)); do
  on $r "p=\$(cat $G6_HOME/out/g6m4.pid 2>/dev/null); [ -n \"\$p\" ] && kill -0 \$p 2>/dev/null && kill \$p; echo rank $r stopped" &
done
wait
