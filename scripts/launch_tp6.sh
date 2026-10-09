#!/bin/bash
# ONE command: boot the K1H2 engine on six Sparks and serve an OpenAI-compatible API on the head (rank 0).
#   scripts/launch_tp6.sh            (after scripts/stage.sh; reads config/cluster.env)
# Starts, in order: g6m4 on every node (ranks 5..0, detached) -> waits for all six serve/ready files ->
# g6shim.py (engine socket, drives the six ranks through serve/ files) -> the front end on :$PORT.
# Stop with scripts/stop.sh. Logs: $G6_HOME/out/ on each node (engine), $G6_HOME/out/{shim,front}.log on the head.
#
# Knobs (environment):
#   CAP=614400        KV pool / max context in tokens (the K1H2 config; lower it if a node runs short of memory)
#   MTP=2             speculative draft length k (2 = our prose/served default; 4-6 helped code/structured)
#   NOHASH=1          skip the sha256 check of every pack tensor at load (faster boot; run it once without first)
#   SOURCES_ROOT=DIR  sparse packs (packs/rank<r>.sources, convert.py --v2-refs): where the v2 files are on the
#                     nodes (default: the root rank<r>.sources names)
#   G6ENV_X="A=1 B=2" extra engine env for every rank (see docs/SETTINGS.md)
#   G6ARGS_X="--top 20" extra engine arguments for every rank (--top K prints the per-slice top-K lines for requests with logprobs)
#   READY_TMO=900     seconds to wait for all six ranks to load
#   FRONT=rs          front end: rs = g6front-rs (Rust, issue #15; the default, built by scripts/build_front.sh; without
#                     it the launcher falls back to py), py = frontend/g6front (Python). Same flags, same API;
#                     FRONT_BIN=path overrides $REPO/bin/g6front-rs
#   RANK_HEALTH=0     skip the rank health probe (default: probe every rank first and refuse to boot, exit 3, with a named
#                     reason per refused rank; see tools/health/README.md). G6_PROGRESS_ISOLATE=0 in G6ENV_X turns off the
#                     engine's move of other threads off the proxy CPUs.
set -eu
. "$(dirname "$0")/common.sh"
[ "$NRANK" = 6 ] || { echo "launch_tp6.sh needs 6 HOSTS, config has $NRANK"; exit 2; }
CAP=${CAP:-614400}; MTP=${MTP:-2}; READY_TMO=${READY_TMO:-900}

# Boot admission: probe every rank (link rate per rail, GPU clock and throttle, wedge, memory rates, proxy CPUs) while no
# model is booted, and stop here when a rank fails. No engine has started yet, so nothing needs cleaning up.
if [ "${RANK_HEALTH:-1}" != 0 ]; then
  "$(dirname "$0")/rank_health.sh" || { rc=$?; echo "boot refused (rank health, exit $rc); fix the named rank, or RANK_HEALTH=0 to skip" >&2; exit "$rc"; }
else
  rm -f "$G6_HOME/out/rank_health.json"      # a result left from an earlier boot must not stand for this one (gate.py run --health)
fi

# K1H2 = the served prefill/decode env + OWN=3 (QO attention) + context parallel KV (CP) past R=32768 tokens
# with the HASH2 staged-page check. Every variable is explained in docs/SETTINGS.md.
ENGINE_ENV="G6_PF_TILE=64 G6_PF_DENSE=8192 G6_PF_RS=1 G6_PF_STREAMS=2 G6_PF_ATTN_ROWS=1024 G6_PF_OWN=3 \
G6_PF_BLAS_ROWS=1024 G6_PF_TIMES=1 G6_PF_HASH_GPU=1 G6_PF_RSDIRECT=1 G6_DEC_SPLIT=3 G6_DEC_SHARED_SCRATCH=1 \
G6_CP=1 G6_CP_R=32768 G6_CP_S3=1 G6_CP_PF_VERIFY=1 G6_CP_PF_HASH2=1 ${G6ENV_X:-}"
SERVE_ARGS="--top-p-cands 256 --chunk 8 --temp 0.7 --cap $CAP --graph --wide --mtp $MTP --prefill 4096 --pf-lanes 2 --pfv2 \
--qo-manifest packs/qo.manifest.json --qo-pack packs/qo.pack"
[ "${NOHASH:-0}" = 1 ] && SERVE_ARGS="$SERVE_ARGS --no-hash"
SERVE_ARGS="$SERVE_ARGS ${G6ARGS_X:-}"

for ((r = NRANK - 1; r >= 0; r--)); do
  # a sparse pack (convert.py --v2-refs) reads its exlr8 v2 expert bytes from the v2 files at load
  SRCARGS=""
  on $r "test -e $G6_HOME/packs/rank$r.sources" && SRCARGS="--sources packs/rank$r.sources ${SOURCES_ROOT:+--sources-root $SOURCES_ROOT}"
  # each process writes its own pid (setsid may fork, so $! is not always the engine's pid)
  on $r "cd $G6_HOME && rm -rf serve && mkdir -p serve out && { \
    nohup setsid bash -c 'echo \$\$ > out/g6m4.pid; exec env $ENGINE_ENV ./bin/g6m4 --rank $r --ips $IPS --ips2 $IPS2 \
      --proxy-cpu $PROXY_CPU --manifest packs/rank$r.manifest.json --pack packs/rank$r.pack --image --layers all $SRCARGS \
      --serve $G6_HOME/serve $SERVE_ARGS' > out/g6m4.r$r.txt 2> out/g6m4.r$r.err < /dev/null & }" &
done
wait
echo "engines started; waiting for six ready files (up to ${READY_TMO}s)"
t0=$(date +%s)
while :; do
  c=0; for ((r = 0; r < NRANK; r++)); do on $r "test -e $G6_HOME/serve/ready" && c=$((c + 1)); done
  [ $c = $NRANK ] && break
  for ((r = 0; r < NRANK; r++)); do
    on $r "p=\$(cat $G6_HOME/out/g6m4.pid 2>/dev/null); [ -z \"\$p\" ] || kill -0 \$p 2>/dev/null" || { echo "rank $r exited; tail of its log:"; on $r "tail -20 $G6_HOME/out/g6m4.r$r.err"; exit 1; }
  done
  [ $(( $(date +%s) - t0 )) -lt "$READY_TMO" ] || { echo "timeout: $c/$NRANK ranks ready"; exit 1; }
  sleep 5
done
echo "all ranks ready after $(( $(date +%s) - t0 ))s"

# shim + front end on the head (rank 0). The head's own serve dir is local; the others are host:path over ssh.
DIRS="$G6_HOME/serve"; for ((r = 1; r < NRANK; r++)); do DIRS="$DIRS ${HOST[$r]}:$G6_HOME/serve"; done
PY=${PY:-$G6_HOME/venv/bin/python}
# shim and front end stay off the proxy CPUs (the engine pins its proxy threads to PROXY_CPU-4..PROXY_CPU on each node)
PIN=""; if [ "$PROXY_CPU" -ge 5 ] && command -v taskset > /dev/null; then PIN="taskset -c 0-$((PROXY_CPU - 5))"; fi
mkdir -p "$G6_HOME/run" "$G6_HOME/out"; SOCK=$G6_HOME/run/engine.sock; rm -f "$SOCK"
PYTHONPATH="$REPO/frontend" nohup setsid bash -c 'echo $$ > "$0"; exec "$@"' "$G6_HOME/out/shim.pid" \
  $PIN "$PY" "$REPO/frontend/g6shim.py" --rank-dirs $DIRS --sock "$SOCK" --max-model-len "$CAP" --stall-timeout 2400 \
  > "$G6_HOME/out/shim.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do [ -S "$SOCK" ] && break; sleep 1; done
[ -S "$SOCK" ] || { echo "shim did not open $SOCK"; tail -20 "$G6_HOME/out/shim.log"; exit 1; }
FRONT_RS="${FRONT_BIN:-$REPO/bin/g6front-rs}"
if [ "${FRONT:-rs}" = rs ] && [ ! -x "$FRONT_RS" ]; then
  [ "${FRONT:-}" = rs ] && { echo "FRONT=rs: $FRONT_RS missing (scripts/build_front.sh)"; exit 1; }
  echo "no $FRONT_RS (scripts/build_front.sh): using the Python front end"
  FRONT=py
fi
if [ "${FRONT:-rs}" = rs ]; then
  FRONT_CMD="$FRONT_RS"
else
  FRONT_CMD="$PY -m g6front"
fi
(cd "$REPO/frontend" && { nohup setsid bash -c 'echo $$ > "$0"; exec "$@"' "$G6_HOME/out/front.pid" \
  $PIN $FRONT_CMD --model-dir "$MODEL_DIR" --engine-sock "$SOCK" --port "$PORT" > "$G6_HOME/out/front.log" 2>&1 < /dev/null & })
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 1; done
echo "front end /health: $(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")  ->  http://<head>:$PORT/v1"
