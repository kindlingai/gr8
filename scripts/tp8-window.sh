#!/bin/bash
# One unattended TP8 hybrid window: take the serves down, boot the TP8 test cluster once, chat, then restore the serves.
#
#   tp8-window.sh run CONF       the primary (start it under systemd-run so it outlives the session)
#   tp8-window.sh restore RD     the recovery owner, on RECOVERY_HOST only (the timer or the primary starts it)
#
# CONF (scripts/tp8-window.conf.example) holds every host, path and hash. The primary never restores: it arms a timer on
# the recovery host before any change, and on success, failure or a signal it starts that host's restore and waits.
# restore is single-owner (flock), fences further TP8 launches, stops the test cluster box by box (down --local and
# verify-down --kill: no cluster-wide down, so no stale epoch-free RPC can reach a restored serve), leaves a serve that
# is already healthy with its canary hash alone, and otherwise restarts it (max 2 attempts). Its end state is
# RD/RESTORED_OK or RD/ALERT. An ALERT is not a restored serve: an unreachable host can still leave a serve down.
#
# A serve is a gr8 glm53 serve unless its conf sets hooks (any serve, e.g. a non-gr8 one): <S>_DOWN_CMD (head, P1),
# <S>_DOWN_S (its timeout, default 300), <S>_STOP_CMD (each box: stop and verify down, exit 0 = down), <S>_UP_CMD (each of <S>_UP_BOXES, must return at once),
# <S>_PREP (each box before up), <S>_PRE_UP_CHECK and <S>_POST_CHECK (each box, exit 0 = pass; PRE before every up,
# POST after the canary), <S>_P0_CHECK (each box in P0, while the serve still runs), <S>_ENV_BOXES (the boxes that hold
# <S>_ENV; default <S>_BOXES), <S>_HEALTH (path, default /health), <S>_HEALTH_S, and <S>_CANARY_MD5=exit0 (the canary
# passes on the command's exit status; no output file; <S>_P0_CANARY_CMD replaces it in P0, e.g. while the serve runs
# a test config that the restore must not reproduce).
# Optional report-only speed rows: BASE_SPEED_CMD on BASE_SPEED_HOST at the end of P0 (live serve), SPEED_CMD on the
# TP8 head after a P6 PASS when at most SPEED_MAX_ELAPSED s have passed (bounded by SPEED_S, added to T_MAX).
set -u
ME=$(readlink -f "$0")

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "${LOG:-/dev/stderr}" >&2; }
die() { say "ABORT: $*"; exit 1; }

# rsh HOST SECONDS CMD: bounded on both ends; the remote side runs CMD under its own timeout
rsh() {
  local h=$1 t=$2; shift 2
  timeout $((t + 20)) ssh -n -o ConnectTimeout=10 -o BatchMode=yes "$SSH_USER@$h" "timeout $t bash -c $(printf %q "$*")"
}
# rput HOST FILE DEST: copy one file
rput() { timeout 60 ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_USER@$1" "cat > $(printf %q "$3")" < "$2"; }
# sv NAME KEY: a serve's setting (A_HOME ...); svboxes NAME: its boxes
sv() { local v="${1}_$2"; printf '%s' "${!v}"; }
svboxes() { local v="${1}_BOXES[@]"; printf '%s\n' "${!v}"; }
# svd NAME KEY DEFAULT: a serve's optional setting
svd() { local v="${1}_$2"; printf '%s' "${!v:-$3}"; }
# svlist NAME KEY: the serve's optional box list <S>_KEY, else <S>_BOXES
svlist() { local v="${1}_$2[@]"; if declare -p "${1}_$2" > /dev/null 2>&1; then printf '%s\n' "${!v}"; else svboxes "$1"; fi; }
svupboxes() { svlist "$1" UP_BOXES; }
envkey() { rsh "$1" 30 "sed -n 's/^\\(export \\)\\{0,1\\}$3=//p' $(printf %q "$2") | tail -1 | tr -d \"\\\"'\""; }

load() {
  # shellcheck source=/dev/null
  . "$1" || die "cannot read $1"
  local k
  for k in SSH_USER RECOVERY_HOST RECOVERY_ROOT SYSTEMD_RUN ALERT_CMD TP8_HOME TP8_ENV TP8_URL TP8_LIFE TP8_READY \
           MEMCLEAN MEMCLEAN_MD5 MEMCLEAN_PATH PREP_CMD VERIFY_WEIGHTS_CMD ERR_RE B_P0 B_ARM B_P1 B_P2 B_P3 B_P4 B_P5 \
           B_P6 B_RESTORE B_MARGIN; do
    [ -n "${!k:-}" ] || die "$1: $k is not set"
  done
  [ "${#TP8_BOXES[@]}" -eq 8 ] || die "$1: TP8_BOXES needs 8 boxes"
  [ "${#SERVES[@]}" -ge 1 ] || die "$1: SERVES is empty"
  for s in "${SERVES[@]}"; do
    for k in HEAD HOME ENV ENV_SHA URL UP CANARY_HOST CANARY_CMD CANARY_MD5; do
      [ -n "$(sv "$s" "$k")" ] || die "$1: ${s}_$k is not set"
    done
    [ "$(svboxes "$s" | wc -l)" -ge 1 ] || die "$1: ${s}_BOXES is empty"
    [ "$(sv "$s" CANARY_MD5)" = exit0 ] || [ -n "$(sv "$s" CANARY_OUT)" ] || die "$1: ${s}_CANARY_OUT is not set"
  done
  T_MAX=$((B_P0 + B_ARM + B_P1 + B_P2 + B_P3 + B_P4 + B_P5 + B_P6 + ${SPEED_CMD:+${SPEED_S:-600} +} B_RESTORE + B_MARGIN))
}

# each BOX in parallel: CMD; prints "BOX rc" lines into $2-file and returns the number of failures
on_all() {
  local t=$1 tag=$2 cmd=$3; shift 3
  local h n=0 d
  d=$(mktemp -d)
  for h in "$@"; do (rsh "$h" "$t" "$cmd" > "$d/$h.out" 2>&1; echo $? > "$d/$h.rc") & done
  wait
  for h in "$@"; do
    sed "s/^/[$tag $h] /" "$d/$h.out" >> "$LOG"
    if [ "$(cat "$d/$h.rc")" != 0 ]; then say "$tag: $h failed (rc $(cat "$d/$h.rc"))"; n=$((n + 1)); fi
  done
  rm -rf "$d"
  return "$n"
}

healthy() { rsh "$1" 30 "curl -s -o /dev/null -m 10 -w '%{http_code}' $(printf %q "$2")${3:-/health}" 2>/dev/null | grep -qx 200; }
shealthy() { healthy "$(sv "$1" HEAD)" "$(sv "$1" URL)" "$(svd "$1" HEALTH /health)"; }
# canary NAME: run the pinned canary command, hash its pinned output file, compare
canary() {   # NAME [KEY]: KEY = CANARY_CMD (default) or P0_CANARY_CMD (exit0 serves only; falls back to CANARY_CMD)
  local s=$1 h md5 key=${2:-CANARY_CMD}
  h=$(sv "$s" CANARY_HOST)
  if [ "$(sv "$s" CANARY_MD5)" = exit0 ]; then
    rsh "$h" 900 "$(svd "$s" "$key" "$(sv "$s" CANARY_CMD)")" >> "$LOG" 2>&1 || { say "$s canary: command failed"; return 1; }
    say "$s canary: exit 0 OK"; return 0
  fi
  rsh "$h" 900 "rm -f $(sv "$s" CANARY_OUT) && $(sv "$s" CANARY_CMD)" >> "$LOG" 2>&1 || { say "$s canary: command failed"; return 1; }
  md5=$(rsh "$h" 30 "md5sum $(sv "$s" CANARY_OUT)" | cut -d' ' -f1)
  [ "$md5" = "$(sv "$s" CANARY_MD5)" ] || { say "$s canary: md5 $md5, want $(sv "$s" CANARY_MD5)"; return 1; }
  say "$s canary: md5 OK"
}
tp8_local_stop() {   # one box: this box's test processes, by recorded identity
  local c
  c="cd $(printf %q "$TP8_HOME") && ./gr8 glm53 down --local @$TP8_ENV; ./gr8 glm53 verify-down --kill @$TP8_ENV"
  on_all 180 tp8-stop "$c" "${TP8_BOXES[@]}"
}
serve_local_stop() {
  local s=$1 home env
  home=$(printf %q "$(sv "$s" HOME)"); env=$(sv "$s" ENV)
  # shellcheck disable=SC2046
  on_all 180 "$s-stop" "$(svd "$s" STOP_CMD "cd $home && ./gr8 glm53 down --local @$env; ./gr8 glm53 verify-down --kill @$env")" \
    $(svboxes "$s")
}
serve_check() {   # NAME KEY: the serve's optional per-box check; unset passes
  local c
  c=$(svd "$1" "$2" "")
  [ -z "$c" ] && return 0
  # shellcheck disable=SC2046
  on_all 300 "$1-${2,,}" "$c" $(svboxes "$1")
}
serve_up() {
  local s=$1 h c
  c=$(svd "$s" UP_CMD \
      "cd $(printf %q "$(sv "$s" HOME)") && setsid nohup timeout 3000 ./gr8 glm53 up @$(sv "$s" ENV) > out/window-up.log 2>&1 < /dev/null &")
  if [ "$(sv "$s" UP)" = together ]; then
    # shellcheck disable=SC2046
    on_all 60 "$s-up" "$c" $(svupboxes "$s")
  else
    for h in $(svupboxes "$s"); do rsh "$h" 60 "$c" >> "$LOG" 2>&1 || say "$s-up: $h failed"; sleep "$(sv "$s" GAP)"; done
  fi
}
wait_health() {   # NAME SECONDS
  local end=$((SECONDS + $2))
  while [ $SECONDS -lt $end ]; do shealthy "$1" && return 0; sleep 30; done
  return 1
}

# ---------------------------------------------------------------- restore (recovery host only)
restore_serve() {   # NAME -> 0 restored and verified
  local s=$1 a
  if shealthy "$s" && canary "$s"; then say "$s: already serving with its canary: left alone"; return 0; fi
  for a in 1 2; do
    say "$s: restore attempt $a"
    serve_local_stop "$s" || { say "$s: not verified down"; continue; }
    # shellcheck disable=SC2046
    on_all "$B_P3" "$s-memclean" "$(svd "$s" PREP "$MEMCLEAN && $PREP_CMD")" $(svboxes "$s") \
      || { say "$s: memclean/prep failed"; continue; }
    serve_check "$s" PRE_UP_CHECK || { say "$s: pre-up check failed"; continue; }
    serve_up "$s"
    if wait_health "$s" "$(svd "$s" HEALTH_S "${SERVE_HEALTH_S:-2400}")" && canary "$s" && serve_check "$s" POST_CHECK; then
      say "$s: restored"; return 0
    fi
  done
  return 1
}

cmd_restore() {
  trap - EXIT INT TERM HUP
  RD=$(readlink -f "$1")
  LOG=$RD/restore.log
  exec 9> "$RD/restore.lock"
  flock -n 9 || { say "restore: another restore owns $RD"; exit 0; }
  [ -e "$RD/RESTORED_OK" ] && { say "restore: already RESTORED_OK"; exit 0; }
  [ -e "$RD/ALERT" ] && { say "restore: ALERT already raised; no automatic retry"; exit 0; }
  (cd "$RD" && sha256sum -c --quiet SHA256SUMS) || { echo "restore: $RD copies do not match their sha256" > "$RD/ALERT"; exit 1; }
  load "$RD/window.conf"
  RUN_ID=$(cat "$RD/RUN_ID")
  touch "$RD/FENCE"
  say "restore $RUN_ID: fenced; stopping the test cluster box by box"
  local s fail=0 tp8ok=1
  tp8_local_stop || tp8ok=0
  if [ $tp8ok = 0 ]; then
    fail=1; say "restore: test processes not verified gone on some box; serves are not started over them"
  else
    for s in "${SERVES[@]}"; do (restore_serve "$s"; echo $? > "$RD/serve-$s.rc") & done
    wait
    for s in "${SERVES[@]}"; do [ "$(cat "$RD/serve-$s.rc" 2>/dev/null)" = 0 ] || fail=1; done
  fi
  if [ $fail = 0 ]; then
    touch "$RD/RESTORED_OK"; say "restore $RUN_ID: RESTORED_OK"
    timeout 30 systemctl stop "tp8-restore-$RUN_ID.timer" 2>/dev/null
  else
    local why="restore $RUN_ID failed: see $RD/restore.log"
    echo "$why" > "$RD/ALERT"; say "ALERT: $why"
    (set -- "$why"; eval "$ALERT_CMD") >> "$LOG" 2>&1
  fi
}

# ---------------------------------------------------------------- run (primary)
RD_REMOTE() { rsh "$RECOVERY_HOST" 30 "$*"; }

# the primary's one way out once P1 may have begun: start restore on the recovery host and wait for its end state
finish() {
  trap - EXIT INT TERM HUP
  local why=$1 end
  say "finish: $why; handing over to restore on $RECOVERY_HOST"
  RD_REMOTE "$SYSTEMD_RUN --unit=tp8-restore-now-$RUN_ID /bin/bash $RD/tp8-window.sh restore $RD" >> "$LOG" 2>&1 \
    || say "finish: could not start restore now (the timer still fires at T_MAX)"
  end=$((SECONDS + B_RESTORE + B_MARGIN))
  while [ $SECONDS -lt $end ]; do
    if RD_REMOTE "test -e $RD/RESTORED_OK"; then
      say "RESTORED_OK ($why)"; echo "$why; RESTORED_OK" > "$LOCAL/RESULT"
      [ "$why" = "TP8 PASS" ] && exit 0; exit 1
    fi
    if RD_REMOTE "test -e $RD/ALERT"; then say "ALERT from restore ($why)"; echo "$why; ALERT" > "$LOCAL/RESULT"; exit 3; fi
    sleep 60
  done
  say "ALERT: no restore end state after $((B_RESTORE + B_MARGIN)) s"
  (set -- "primary saw no restore end state for $RUN_ID"; eval "$ALERT_CMD") >> "$LOG" 2>&1
  echo "$why; NO END STATE" > "$LOCAL/RESULT"
  exit 3
}

p0() {
  say "P0 preflight (read only)"
  local all=("${TP8_BOXES[@]}") s h p ports=() deny cp scp
  for s in "${SERVES[@]}"; do mapfile -t -O "${#all[@]}" all < <(svboxes "$s"); done
  mapfile -t all < <(printf '%s\n' "${all[@]}" "$RECOVERY_HOST" | sort -u)
  on_all 30 ssh true "${all[@]}" || die "ssh to every box"
  on_all 30 recovery-ssh "for b in ${all[*]}; do timeout 20 ssh -n -o ConnectTimeout=10 -o BatchMode=yes $SSH_USER@\$b true || exit 1; done" \
    "$RECOVERY_HOST" || die "the recovery host cannot ssh to every box"
  RD_REMOTE "command -v flock && command -v $SYSTEMD_RUN" > /dev/null || die "the recovery host lacks flock or systemd-run"
  for s in "${SERVES[@]}"; do
    shealthy "$s" || die "$s is not serving"
    canary "$s" P0_CANARY_CMD || die "$s baseline canary"
    serve_check "$s" P0_CHECK || die "$s P0 check (its restore would fail)"
    for h in $(svlist "$s" ENV_BOXES); do
      [ "$(rsh "$h" 30 "sha256sum $(sv "$s" ENV)" | cut -d' ' -f1)" = "$(sv "$s" ENV_SHA)" ] || die "$s env on $h changed"
    done
  done
  local sums=()
  for h in "${TP8_BOXES[@]}"; do sums+=("$(rsh "$h" 60 "cd $TP8_HOME && md5sum bin/* | md5sum" | cut -d' ' -f1)"); done
  [ "$(printf '%s\n' "${sums[@]}" | sort -u | wc -l)" = 1 ] && [ -n "${sums[0]}" ] || die "TP8 bin/ differs or is missing: ${sums[*]}"
  on_all 60 memclean-md5 "[ \$(md5sum < $MEMCLEAN_PATH | cut -d' ' -f1) = $MEMCLEAN_MD5 ]" "${TP8_BOXES[@]}" || die "memclean md5"
  on_all 30 dirty "[ \$(awk '/^Dirty:/{print \$2}' /proc/meminfo) -lt ${DIRTY_MAX_KB:-1048576} ]" "${TP8_BOXES[@]}" || die "dirty pages"
  [ -n "${P0_BOX_CHECK:-}" ] && { on_all 600 box-check "$P0_BOX_CHECK" "${TP8_BOXES[@]}" || die "P0_BOX_CHECK"; }
  # isolation: the test control port is set and is no serve's (unset in a serve env = 7340)
  cp=$(envkey "${TP8_BOXES[0]}" "$TP8_ENV" CONTROL_PORT)
  [ -n "$cp" ] || die "the TP8 env sets no CONTROL_PORT"
  for s in "${SERVES[@]}"; do
    scp=$(envkey "$(sv "$s" HEAD)" "$(sv "$s" ENV)" CONTROL_PORT); scp=${scp:-7340}
    [ "$cp" != "$scp" ] || die "TP8 CONTROL_PORT $cp is $s's control port"
  done
  for h in "${TP8_BOXES[@]}"; do
    p=$(envkey "$h" "$TP8_ENV" FAB_PORTS); [ -n "$p" ] || die "$h: no FAB_PORTS"; ports+=("$p")
  done
  deny=$(envkey "${TP8_BOXES[0]}" "$TP8_ENV" FAB_DENY)
  ROUTES_SHA=$(rsh "${TP8_BOXES[0]}" 60 "python3 $TP8_HOME/tools/launch/routes.py ${ports[*]} ${deny:+--deny $deny}" \
               | tee -a "$LOG" | sed -n 's/^routes sha \([0-9a-f]\{64\}\)$/\1/p')
  [ -n "$ROUTES_SHA" ] || die "routes.py dry run refused the declared ports"
  echo "$ROUTES_SHA" > "$LOCAL/routes.sha"
  if [ -n "${BASE_SPEED_CMD:-}" ]; then   # report only: the same speed row on a live serve, for compare
    say "P0 base speed row on $BASE_SPEED_HOST (report only)"
    rsh "$BASE_SPEED_HOST" "${BASE_SPEED_S:-600}" "$BASE_SPEED_CMD" > "$LOCAL/speed-base.txt" 2>&1 || say "P0: base speed row failed (not a gate)"
  fi
  say "P0 OK: routes sha $ROUTES_SHA, T_MAX ${T_MAX}s"
}

arm() {
  say "arm: recovery dir and timer on $RECOVERY_HOST"
  local st
  RD_REMOTE "mkdir -p $RECOVERY_ROOT && mkdir $RD" || die "$RD exists or cannot be made"
  rput "$RECOVERY_HOST" "$ME" "$RD/tp8-window.sh" || die "copy the script"
  rput "$RECOVERY_HOST" "$CONF" "$RD/window.conf" || die "copy the conf"
  RD_REMOTE "echo $RUN_ID > $RD/RUN_ID && cd $RD && sha256sum tp8-window.sh window.conf > SHA256SUMS" || die "write ids"
  [ "$(RD_REMOTE "cat $RD/SHA256SUMS" | awk '{print $1}' | tr '\n' ' ')" = "$(sha256sum "$ME" "$CONF" | awk '{print $1}' | tr '\n' ' ')" ] \
    || die "the copies on $RECOVERY_HOST do not match"
  RD_REMOTE "$SYSTEMD_RUN --unit=tp8-restore-$RUN_ID --on-active=${T_MAX}s /bin/bash $RD/tp8-window.sh restore $RD" >> "$LOG" 2>&1 \
    || die "arm the timer"
  st=$(RD_REMOTE "systemctl show -p ActiveState --value tp8-restore-$RUN_ID.timer")
  [ "$st" = active ] || die "timer tp8-restore-$RUN_ID.timer is '$st', not active"
  say "armed: tp8-restore-$RUN_ID.timer fires in ${T_MAX}s"
}

fenced() { RD_REMOTE "test -e $RD/FENCE"; }

cmd_run() {
  CONF=$(readlink -f "$1")
  load "$CONF"
  RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM
  RD=$RECOVERY_ROOT/$RUN_ID
  LOCAL=$LOG_ROOT/$RUN_ID
  mkdir -p "$LOCAL" || exit 1
  LOG=$LOCAL/window.log
  say "run $RUN_ID: conf $CONF"
  p0
  arm
  trap 'finish "signal"' INT TERM HUP
  trap 'finish "unexpected exit"' EXIT
  local s ok=1 h shas

  say "P1 serves down"
  fenced && finish "fenced before P1"
  for s in "${SERVES[@]}"; do
    rsh "$(sv "$s" HEAD)" "$(svd "$s" DOWN_S 300)" "$(svd "$s" DOWN_CMD "cd $(sv "$s" HOME) && ./gr8 glm53 down @$(sv "$s" ENV)")" >> "$LOG" 2>&1
    serve_local_stop "$s" || finish "P1: $s not verified down"
  done

  say "P2 verify weights"
  on_all "$B_P2" weights "$VERIFY_WEIGHTS_CMD" "${TP8_BOXES[@]}" || finish "P2: weights"

  say "P3 memclean and prep"
  on_all "$B_P3" prep "$MEMCLEAN && $PREP_CMD" "${TP8_BOXES[@]}" || finish "P3: memclean/prep"

  say "P4 TP8 up"
  fenced && finish "fenced before P4: restore has begun"
  on_all 60 tp8-up "cd $TP8_HOME && setsid nohup timeout $TP8_LIFE ./gr8 glm53 up @$TP8_ENV > out/window-up.log 2>&1 < /dev/null &" \
    "${TP8_BOXES[@]}" || finish "P4: launch"

  say "P5 wait for /health (<= ${B_P5}s), error scan every 60 s"
  local end=$((SECONDS + B_P5)) up=0
  while [ $SECONDS -lt $end ]; do
    if ! on_all 30 errscan "! grep -E -H -m3 $(printf %q "$ERR_RE") $TP8_HOME/out/g6m4.r*.err" "${TP8_BOXES[@]}"; then
      finish "P5: fabric error in the engine logs"
    fi
    healthy "${TP8_BOXES[0]}" "$TP8_URL" && { up=1; break; }
    sleep 60
  done
  [ $up = 1 ] || finish "P5: no /health within ${B_P5}s"

  say "P6 two chats at temperature 0"
  local body
  body=$(printf '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"temperature":0,"max_tokens":%s}' "${TP8_CHAT_TOKENS:-128}")
  for h in 1 2; do
    rsh "${TP8_BOXES[0]}" 600 "curl -s -m 580 -H 'Content-Type: application/json' -d $(printf %q "$body") $TP8_URL/v1/chat/completions" \
      > "$LOCAL/chat$h.json" 2>> "$LOG" || ok=0
    python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["choices"][0]; t=c["message"]["content"]; assert t and c.get("finish_reason"); open(sys.argv[2],"w").write(t)' \
      "$LOCAL/chat$h.json" "$LOCAL/chat$h.txt" 2>> "$LOG" || ok=0
  done
  cmp -s "$LOCAL/chat1.txt" "$LOCAL/chat2.txt" || { say "P6: the two temperature-0 chats differ"; ok=0; }
  healthy "${TP8_BOXES[0]}" "$TP8_URL" || ok=0
  for h in "${TP8_BOXES[@]}"; do
    rsh "$h" 30 "grep -o 'routes sha [0-9a-f]*' $TP8_HOME/out/supervisor.log | tail -1; sha256sum $TP8_HOME/run/routes.txt" \
      > "$LOCAL/routes.$h" 2>> "$LOG"
  done
  shas=$(cat "$LOCAL"/routes.* | grep -oE '[0-9a-f]{64}' | sort | uniq -c)
  echo "$shas" >> "$LOG"
  [ "$(echo "$shas" | awk -v s="$ROUTES_SHA" '$2 == s {print $1}')" = 16 ] && [ "$(echo "$shas" | wc -l)" = 1 ] \
    || { say "P6: the 8 boxes' routes sha (log + run/routes.txt) are not all $ROUTES_SHA"; ok=0; }
  for h in "${TP8_BOXES[@]}"; do   # best effort, bounded
    mkdir -p "$LOCAL/logs/$h"
    timeout 120 scp -q -o ConnectTimeout=10 -o BatchMode=yes "$SSH_USER@$h:$TP8_HOME/out/*.log" "$SSH_USER@$h:$TP8_HOME/out/*.err" \
      "$LOCAL/logs/$h/" 2>> "$LOG"
  done
  if [ $ok = 1 ] && [ -n "${SPEED_CMD:-}" ]; then   # P7, report only, never a gate; skipped late in the window
    if [ $SECONDS -le "${SPEED_MAX_ELAPSED:-4500}" ] && ! fenced; then
      say "P7 speed row (report only, <= ${SPEED_S:-600}s)"
      rsh "${TP8_BOXES[0]}" "${SPEED_S:-600}" "$SPEED_CMD" > "$LOCAL/speed-tp8.txt" 2>&1 || say "P7: speed row failed (not a gate)"
    else
      say "P7: speed row skipped (${SECONDS}s since start, or fenced)"
    fi
  fi
  if [ $ok = 1 ]; then finish "TP8 PASS"; else finish "TP8 FAIL at P6"; fi
}

case "${1:-}" in
  run) [ $# = 2 ] || { echo "usage: $0 run CONF" >&2; exit 2; }; cmd_run "$2" ;;
  restore) [ $# = 2 ] || { echo "usage: $0 restore RD" >&2; exit 2; }; cmd_restore "$2" ;;
  *) echo "usage: $0 run CONF | restore RD" >&2; exit 2 ;;
esac
