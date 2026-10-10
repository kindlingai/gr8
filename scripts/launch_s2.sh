#!/bin/bash
# Multi-session (S2) boot of the TP4, TP6 or TP3 engine on four, six or three Sparks: g6m4 --serve (scripts/build.sh builds
# every TP with DEC_S2) serving up to SESS concurrent sessions, driven by frontend/g6s2shim.py. Stop with scripts/stop.sh.
#   SESS=8 scripts/launch_s2.sh        (scripts/launch_tp4.sh, launch_tp6.sh and launch_tp3.sh run this)
# Knobs (environment; KNOBS.md has all of them):
#   SESS=8            most concurrent sessions (1..8); every reachable step graph is captured at boot (~2.5 GiB at 8)
#   MTP=2             MTP draft count for sessions that share steps
#   DFLASH=1          also load the DFlash2 drafter: a session alone at its prompt tail drafts with it (adaptive draft
#                     count, SPEC_ADAPT=1) and goes to MTP when a second session arrives. DFLASH=0: MTP only
#   CAP, PACKS, PF_OWN (default 4; short prompts take OWN=2, G6_PF_QP_MIN), QP_ROLL, QP_CTX, QP_EARLY,
#   V2 / V2_SLICES / V2_WIDTHS, NOHASH, G6ENV_X, FRONT_BIN (the Rust front end; S2 has no Python front end path here)
#   S2PORT=7341       the S2 control port on the head
#   TP=6 only:        PF_OWN defaults to 3 (loads packs/qo.pack; PF_OWN=1 needs no QO pack). DFLASH=1 is refused (the
#                     drafter's 8 KV heads do not split six ways). RANK_HEALTH=0 skips the rank health probe, which
#                     runs before any engine starts (default: refuse to boot, exit 3, with a named reason per refused
#                     rank; tools/health/README.md). The shim and front end run on cores 0..PROXY_CPU-8, below the
#                     engine's proxy cores, and the shim errors a request after STALL_TMO=2400 s without a response.
#   SPEC_ACCEPT=rejection  drafts are greedy and accepted by rejection sampling at temperature > 0
#   REUSE=1           --reuse-prefix, the default (REUSE=0 turns it off): a retired session's caches stay in the pool,
#                     and a request that continues its prompt + output re-prefills only the rest (exact: whole 4,096-row
#                     prefill cells, s2/pool.h)
set -eu
. "$(dirname "$0")/common.sh"
[ "$NRANK" = 4 ] || [ "$NRANK" = 6 ] || [ "$NRANK" = 3 ] || { echo "launch_s2.sh needs 4, 6 or 3 HOSTS, config has $NRANK"; exit 2; }
TP=$NRANK
. "$(dirname "$0")/engine_args.sh"
engine_defaults
if [ "$TP" = 6 ]; then
  # Boot admission: probe every rank (link rate per rail, GPU clock and throttle, wedge, memory rates, proxy CPUs) while
  # no model is booted, and stop here when a rank fails. No engine has started yet, so nothing needs cleaning up.
  if [ "${RANK_HEALTH:-1}" != 0 ]; then
    "$(dirname "$0")/rank_health.sh" || { rc=$?; echo "boot refused (rank health, exit $rc); fix the named rank, or RANK_HEALTH=0 to skip" >&2; exit "$rc"; }
  else
    rm -f "$G6_HOME/out/rank_health.json"      # a result left from an earlier boot must not stand for this one (gate.py run --health)
  fi
fi
EPOCH=$(date +%s)
if [ "$DFLASH" = 1 ]; then
  for ((r = 0; r < NRANK; r++)); do
    on $r "test -e $G6_HOME/packs/dflash.r$r.manifest.json -a -e $G6_HOME/packs/dflash.r$r.pack" || {
      echo "rank $r has no DFlash2 drafter pack ($G6_HOME/packs/dflash.r$r.{manifest.json,pack}); stage it or run with DFLASH=0"
      exit 2; }
  done
fi
engine_env   # scripts/engine_args.sh has the engine environment and arguments, shared with ./gr8 glm53 up
for ((r = NRANK - 1; r >= 0; r--)); do
  ARGS=$(engine_args $r "$PACKS" packs "$G6_HOME" "$EPOCH")
  on $r "cd $G6_HOME && rm -rf s2 serve && mkdir -p s2 out && { \
    nohup setsid bash -c 'echo \$\$ > out/g6m4.pid; exec env LD_LIBRARY_PATH=$G6_HOME/bin/lib $ENGINE_ENV ./bin/g6m4 --rank $r --ips $IPS --ips2 $IPS2 \
      --proxy-cpu $PROXY_CPU $ARGS' > out/g6m4.r$r.txt 2> out/g6m4.r$r.err < /dev/null & }" &
done
wait
PY=${PY:-$G6_HOME/venv/bin/python}
mkdir -p "$G6_HOME/run" "$G6_HOME/out"; SOCK=$G6_HOME/run/engine.sock; rm -f "$SOCK"
# At TP=6 the shim and front end stay off the engine's proxy cores (PROXY_CPU-7..PROXY_CPU, one per fabric)
PIN=""; SHIM_ARGS=""
if [ "$TP" = 6 ]; then
  [ "$PROXY_CPU" -ge 8 ] || { echo "PROXY_CPU=$PROXY_CPU < 8: no room for the shim/front band below the proxy cores"; exit 2; }
  command -v taskset > /dev/null && PIN="taskset -c 0-$((PROXY_CPU - 8))"
  SHIM_ARGS="--stall-timeout ${STALL_TMO:-2400}"
fi
FRONT_RS="${FRONT_BIN:-$REPO/bin/g6front-rs}"
[ -x "$FRONT_RS" ] || { echo "$FRONT_RS missing (scripts/build_front.sh)"; exit 1; }
(cd "$REPO/frontend" && { nohup setsid bash -c 'echo $$ > "$0"; exec "$@"' "$G6_HOME/out/front.pid" \
  $PIN "$FRONT_RS" --model-dir "$MODEL_DIR" --engine-sock "$SOCK" --port "$PORT" > "$G6_HOME/out/front.log" 2>&1 < /dev/null & })
echo "engines started (epoch $EPOCH); waiting for ready.s2 (up to ${READY_TMO}s)"
t0=$(date +%s)
while :; do
  on 0 "test -e $G6_HOME/s2/ready.s2" && break
  for ((r = 0; r < NRANK; r++)); do
    on $r "p=\$(cat $G6_HOME/out/g6m4.pid 2>/dev/null); [ -z \"\$p\" ] || kill -0 \$p 2>/dev/null" || { echo "rank $r exited; tail of its log:"; on $r "tail -20 $G6_HOME/out/g6m4.r$r.err"; exit 1; }
  done
  [ $(( $(date +%s) - t0 )) -lt "$READY_TMO" ] || { echo "timeout waiting for ready.s2"; exit 1; }
  sleep 2
done
echo "ready.s2 after $(( $(date +%s) - t0 ))s: $(on 0 "cat $G6_HOME/s2/ready.s2")"
PYTHONPATH="$REPO/frontend" nohup setsid bash -c 'echo $$ > "$0"; exec "$@"' "$G6_HOME/out/shim.pid" \
  $PIN "$PY" "$REPO/frontend/g6s2shim.py" --dir "$G6_HOME/s2" --sock "$SOCK" $SHIM_ARGS > "$G6_HOME/out/shim.log" 2>&1 < /dev/null &
for i in $(seq 1 150); do [ -S "$SOCK" ] && break; sleep 0.2; done
[ -S "$SOCK" ] || { echo "shim did not open $SOCK"; tail -20 "$G6_HOME/out/shim.log"; exit 1; }
for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 0.2; done
echo "front end /health: $(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")"
