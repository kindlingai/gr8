#!/bin/bash
# DRAFT, NOT INSTALLED (2026-10-20). serve-t5xr.sh up|down|status : the T5XR boot option = T5X + a remote KV store on a
# 6th Spark (KV node dgx2, 10.98.0.2:7350, 106 GiB node RAM = 21G per rank). Runs on dgx1, like serve-t5x.sh.
# up:   refuses if any g6m4 serve runs on the five boxes (stop T5X first: serve-t5x.sh down); starts the KV node on dgx2
#       if it is not serving (memclean, unit rkv-node) and waits for 106 GiB free; memclean + 2000 MHz cap on the five;
#       starts the ranks (unit tp5-serve, LimitMEMLOCK for RDMA); waits /health + 617,532 ctx.
# down: ./gr8 glm53 down, waits until no $H process is left, then stops the KV node (its entries are lost: no persist).
#       KEEP_NODE=1 leaves the KV node up.
set -u -o pipefail
BOXES="100.75.70.62 100.117.67.104 100.125.138.17 100.122.2.119 100.107.163.40"   # r0..r4: dgx1 dgx3 dgx5 dgx6 dgx4
H=/root/rkv/gr8ns; E=/root/rkv/T5XR.env; U=tp5-serve; CTX=617532; AB=/root/ablit
NODE=root@10.98.0.2; NH=/root/rkv/gr8; GB=106
on() {
  local b; for b in $BOXES; do
    ssh -n -o ConnectTimeout=10 -o BatchMode=yes root@$b "$1" || { echo "$b: step failed" >&2; return 1; }
  done
}
anyg6='for p in $(pgrep -x "g6m4|g6m4-int4"); do echo "$p $(readlink /proc/$p/exe)"; done'
ours="for p in \$(pgrep -x 'g6m4|g6m4-int4'); do case \"\$(readlink /proc/\$p/exe)\" in $H/*) echo \$p;; esac; done"
health() { [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/health)" = 200 ]; }
ctx() { curl -s -m 5 http://127.0.0.1:8000/v1/models | grep -o '"max_model_len":[0-9]*' | head -1 | cut -d: -f2; }
node() { ssh -n -o ConnectTimeout=10 -o BatchMode=yes $NODE "cd $NH && ./gr8 kv $1" 2>&1; }

node_up() {
  node status | grep -q serving && { echo "KV node already serving: $(node status | sed -n 2p)"; return 0; }
  ssh -n -o BatchMode=yes $NODE "bash $AB/memclean.sh > /dev/null 2>&1; systemctl reset-failed rkv-node 2>/dev/null
    systemd-run --quiet --collect -p LimitMEMLOCK=infinity --unit=rkv-node --setenv=PATH=\$PATH /bin/bash -c 'cd $NH && exec ./gr8 kv up --gb $GB --mentat none --node-ip 10.98.0.2 --foreground > /root/rkv/node-up.log 2>&1'" || return 1
  local i; for i in $(seq 60); do node status | grep -q "$GB.00 GiB free" && { echo "KV node up: $(node status | sed -n 2p)"; return 0; }; sleep 5; done
  echo "KV node not serving $GB GiB after 300 s: $(ssh -n $NODE tail -3 /root/rkv/node-up.log)" >&2; return 1
}

down() {
  (cd $H && timeout 180 ./gr8 glm53 down @$E) || echo "gr8 down rc $? (checking processes)"
  on "for i in \$(seq 45); do [ -z \"\$($ours)\" ] && break; sleep 2; done; systemctl stop $U 2>/dev/null
      p=\$($ours); [ -z \"\$p\" ] || { echo \"\$(hostname): T5XR processes left: \$p\"; exit 1; }" || return 1
  [ -n "${KEEP_NODE:-}" ] || { node down; ssh -n $NODE 'systemctl stop rkv-node 2>/dev/null; true'; }
}

up() {
  local left
  left=$(for b in $BOXES; do ssh -n -o ConnectTimeout=10 -o BatchMode=yes root@$b "$anyg6" | sed "s/^/$b /"; done)
  [ -z "$left" ] || { echo "a g6m4 serve runs (T5X? run serve-t5x.sh down first): $left" >&2; exit 1; }
  on "test -r $E && grep -q '^TP=5\$' $E && grep -q '^KV_STORE=remote:' $E && test -x $H/bin/g6m4 && test -x $H/bin/g6front-rs" || exit 1
  node_up || exit 1
  on "bash $AB/memclean.sh > /dev/null 2>&1 && nvidia-smi -lgc 0,2000 > /dev/null && cpupower idle-set -D 0 > /dev/null" || exit 1
  on "systemctl reset-failed $U 2>/dev/null; mkdir -p $H/out && systemd-run --quiet --collect -p RemainAfterExit=yes -p TimeoutStopSec=30 -p LimitMEMLOCK=infinity --setenv=HOME --setenv=PATH --unit=$U /bin/bash -c 'cd $H && exec ./gr8 glm53 up @$E > out/window-up.log 2>&1 < /dev/null'" || exit 1
  local t0=$(date +%s)
  until health; do
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "no /health after 900 s: $(grep -iE 'error|fail|kv' $H/out/window-up.log | tail -3)" >&2; exit 1; }
    systemctl is-active -q $U || { echo "rank 0 exited: $(tail -3 $H/out/window-up.log)" >&2; exit 1; }
    sleep 2
  done
  [ "$(ctx)" = "$CTX" ] || { echo "serving, but max_model_len $(ctx) != $CTX" >&2; exit 1; }
  echo "T5XR up in $(( $(date +%s) - t0 )) s, max_model_len $CTX; KV node $(node status | sed -n 2p)"
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) node status | sed -n 1,2p; health && echo "health 200, max_model_len $(ctx)" || { echo "not serving"; exit 1; } ;;
  *) echo "usage: $0 up|down|status" >&2; exit 2 ;;
esac
