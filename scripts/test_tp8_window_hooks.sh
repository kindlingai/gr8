#!/bin/bash
# tp8-window.sh test-cluster hooks (TP8_NBOX, TEST_STOP_CMD, TEST_VERIFY_CMD, ROUTES_CHECK=0, STEPS_CMD) against mocks:
# a whole primary run (P0 .. restore) on 4 mock boxes, a hooked serve, and a hooked test cluster. ssh runs the command
# here; curl, systemctl, systemd-run, scp are stubs. No host, no GPU. Run: bash scripts/test_tp8_window_hooks.sh
# shellcheck disable=SC2015,SC2016
set -u
W=$(readlink -f "$(dirname "$0")/tp8-window.sh")
pass=0 failn=0
ok() { echo "ok   $1"; pass=$((pass + 1)); }
bad() { echo "FAIL $1"; failn=$((failn + 1)); }

setup() {
  T=$(mktemp -d); M=$T/mock; mkdir -p "$M/bin" "$T/t/out" "$T/t/bin" "$T/a" "$T/rec"
  export MOCK=$M PATH="$M/bin:$PATH"
  cat > "$M/bin/ssh" <<'EOF'
#!/bin/bash
while [ $# -gt 0 ]; do case $1 in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host=${1#*@}; shift
echo "ssh $host $*" >> "$MOCK/calls"
MOCK_HOST=$host exec bash -c "$*"
EOF
  cat > "$M/bin/curl" <<'EOF'
#!/bin/bash
case "$*" in
  *chat/completions*) echo '{"choices":[{"message":{"content":"Hello."},"finish_reason":"stop"}]}' ;;
  *:8100*) [ -e "$MOCK/tup" ] && printf 200 || printf 000 ;;
  *) [ -e "$MOCK/up" ] && printf 200 || printf 000 ;;
esac
EOF
  cat > "$M/bin/systemctl" <<'EOF'
#!/bin/bash
echo "systemctl $*" >> "$MOCK/calls"
# a stop of the dead-man timer makes it (and its service) inactive, unless timer.stuck
case "$1 ${2:-}" in "stop tp8-restore-"*.timer) [ -e "$MOCK/timer.stuck" ] || touch "$MOCK/timer.stopped" ;; esac
[ "$1" = show ] && { case "$*" in *tp8-restore-*) [ -e "$MOCK/timer.stopped" ] && { echo inactive; exit 0; } ;; esac; echo active; }
# steps.live: the P2b units (tp8-steps-*) run until a stop for that name (with steps.stuck: a stop does not end them)
case "$1 ${2:-}" in
  "stop tp8-steps-"*) [ -e "$MOCK/steps.stuck" ] || touch "$MOCK/stopped.$2" ;;
  "list-units "*) for u in "$@"; do case $u in tp8-steps-*) [ -e "$MOCK/steps.live" ] && [ ! -e "$MOCK/stopped.$u" ] && echo "$u loaded active running x" ;; esac; done ;;
esac
exit 0
EOF
  # systemd-run: a timer (--on-active) is only logged; a unit without one runs now, in the foreground
  cat > "$M/bin/srun" <<'EOF'
#!/bin/bash
echo "srun $*" >> "$MOCK/calls"
case "$*" in *--on-active=*) exit 0 ;; esac
while [ $# -gt 0 ]; do case $1 in --*) shift ;; *) break ;; esac; done
"$@"
EOF
  cat > "$M/bin/detach" <<'EOF'
#!/bin/bash
while [ $# -gt 0 ]; do case $1 in --unit=*) u=${1#--unit=}; shift; break ;; *) shift ;; esac; done
echo "detach $MOCK_HOST $u" >> "$MOCK/calls"
exec bash -c "$3"
EOF
  printf '#!/bin/bash\nexit 0\n' > "$M/bin/scp"
  # md5sum: a bin/ hash is logged (md5bin HOST); with bindiff, b2's bin/ hashes differently
  cat > "$M/bin/md5sum" <<'EOF'
#!/bin/bash
case "$*" in *bin/*) echo "md5bin $MOCK_HOST" >> "$MOCK/calls"; [ -e "$MOCK/md5restore" ] && { c=$(/bin/cat "$MOCK/md5restore"); rm -f "$MOCK/md5restore"; bash -c "$c" > /dev/null 2>&1; }; [ -e "$MOCK/bindiff" ] && [ "$MOCK_HOST" = b2 ] && { echo "ffff  bin/g6m4"; exit 0; } ;; esac
exec /usr/bin/md5sum "$@"
EOF
  printf '#!/bin/bash\necho "tgr8 $MOCK_HOST $*" >> "$MOCK/calls"; touch "$MOCK/tup"\n' > "$T/t/gr8"
  chmod +x "$M"/bin/* "$T/t/gr8"
  echo bin1 > "$T/t/bin/g6m4"
  echo hello > "$M/canary.good"; cp "$M/canary.good" "$M/canary.src"
  echo CONTROL_PORT=7350 > "$T/t.env"; echo PORT=8000 > "$T/a.env"; echo mem > "$T/memclean.sh"
  touch "$M/up"
  cat > "$T/window.conf" <<EOF
SSH_USER=u
RECOVERY_HOST=rec
RECOVERY_ROOT=$T/rec
LOG_ROOT=$T/runs
SYSTEMD_RUN=$M/bin/srun
DETACH=$M/bin/detach
ALERT_CMD='echo "\$1" > $M/alerted'
TP8_NBOX=4
TP8_BOXES=(b0 b1 b2 b3)
TP8_HOME=$T/t
TP8_ENV=$T/t.env
TP8_URL=http://b0:8100
TP8_LIFE=60
TP8_READY=60
TEST_STOP_CMD='echo "tstop \$MOCK_HOST" >> $M/calls; rm -f $M/tup'
TEST_VERIFY_CMD='echo "tverify \$MOCK_HOST" >> $M/calls'
ROUTES_CHECK=0
SERVES=(A)
A_BOXES=(b0 b1 b2 b3)
A_HEAD=b0
A_HOME=$T/a
A_ENV=$T/a.env
A_ENV_SHA=$(sha256sum < "$T/a.env" | cut -d' ' -f1)
A_URL=http://b0:8000
A_UP=together
A_GAP=0
A_DOWN_CMD='echo "adown \$MOCK_HOST" >> $M/calls; rm -f $M/up'
A_STOP_CMD='echo "astop \$MOCK_HOST" >> $M/calls'
A_UP_CMD='echo "aup \$MOCK_HOST" >> $M/calls; touch $M/up'
A_CANARY_HOST=b0
A_CANARY_CMD='cp $M/canary.src $M/canary.out'
A_CANARY_OUT=$M/canary.out
A_CANARY_MD5=$(md5sum < "$M/canary.good" | cut -d' ' -f1)
MEMCLEAN='echo "memclean \$MOCK_HOST" >> $M/calls'
MEMCLEAN_MD5=$(md5sum < "$T/memclean.sh" | cut -d' ' -f1)
MEMCLEAN_PATH=$T/memclean.sh
PREP_CMD=true
VERIFY_WEIGHTS_CMD=true
ERR_RE=FAIL
SERVE_HEALTH_S=5
B_P0=1; B_ARM=1; B_P1=780; B_P2=1; B_P3=30; B_P4=1; B_P5=5; B_P6=1; B_RESTORE=30; B_MARGIN=1
EOF
  : > "$M/calls"
}
run() { bash "$W" run "$T/window.conf" > "$T/out" 2>&1; echo $? > "$T/rc"; RID=$(ls "$T/runs" 2>/dev/null | head -1); }
steps() { cat >> "$T/window.conf" <<EOF
STEPS_HOST=b0
B_STEPS=20
STEPS_FENCE_DIR=$T/fence
STEPS_CMD='echo "steps \$MOCK_HOST" >> $M/calls; ${1:-true}'
EOF
}
first() { grep -n -m1 -F "$1" "$M/calls" | cut -d: -f1; }

# 1. TP8_NBOX: 4 boxes accepted with TP8_NBOX=4, refused with the default 8
setup; bash "$W" cmds "$T/window.conf" > "$T/out" 2>&1 && ok "TP8_NBOX=4, 4 boxes: accepted" || bad "TP8_NBOX=4, 4 boxes: accepted"
sed -i '/^TP8_NBOX=/d' "$T/window.conf"
bash "$W" cmds "$T/window.conf" > "$T/out" 2>&1; [ $? != 0 ] && grep -q 'TP8_BOXES needs 8 boxes' "$T/out" \
  && ok "default 8, 4 boxes: refused" || bad "default 8, 4 boxes: refused"

# 2. STEPS_CMD needs STEPS_HOST and B_STEPS; B_STEPS is in T_MAX
setup; echo "STEPS_CMD=true" >> "$T/window.conf"
bash "$W" cmds "$T/window.conf" > "$T/out" 2>&1; [ $? != 0 ] && grep -q 'STEPS_CMD needs STEPS_HOST and B_STEPS' "$T/out" \
  && ok "STEPS_CMD without host/bound: refused" || bad "STEPS_CMD without host/bound: refused"
setup; t0=$(TP8_WINDOW_LIB=1 bash -c ". '$W'; load '$T/window.conf'; echo \$T_MAX"); steps
t1=$(TP8_WINDOW_LIB=1 bash -c ". '$W'; load '$T/window.conf'; echo \$T_MAX")
[ "$t1" = $((t0 + 20)) ] && ok "B_STEPS added to T_MAX ($t0 -> $t1)" || bad "B_STEPS added to T_MAX ($t0 -> $t1)"

# 3. cmds: the test stop is the unit stop then TEST_STOP_CMD; the steps row is printed
setup; steps; bash "$W" cmds "$T/window.conf" > "$T/cmds" 2>&1
grep -qP '^tp8-stop\tb0 b1 b2 b3\t\( u=.tp8-up-RUNID.service.*\) \|\| exit 1; echo "tstop' "$T/cmds" && ! grep -q 'launch.py down --local' "$T/cmds" \
  && ok "cmds: unit stop then TEST_STOP_CMD, no launch.py" || bad "cmds: unit stop then TEST_STOP_CMD, no launch.py"
grep -qP '^tp8-p0-verify\tb0 b1 b2 b3\techo "tverify' "$T/cmds" && ok "cmds: TEST_VERIFY_CMD" || bad "cmds: TEST_VERIFY_CMD"
grep -qP '^tp8-steps\tb0\t.*--unit=tp8-steps-RUNID /bin/bash -c .*export\\ TP8_RUN_ID=RUNID.*/tmp/tp8-steps-RUNID.rc' "$T/cmds" && ok "cmds: steps row = detached unit tp8-steps-<run>" || bad "cmds: steps row = detached unit tp8-steps-<run>"
grep -qP "^tp8-steps-ctl-stop\tb0\tu='tp8-steps-RUNID.service'" "$T/cmds" && grep -qP "^tp8-steps-stop\tb0 b1 b2 b3\tu='tp8-steps-RUNID-\\*.service'" "$T/cmds" \
  && ok "cmds: steps controller stop (STEPS_HOST) + per-box steps unit stop (all boxes)" || bad "cmds: steps controller stop + per-box steps unit stop"

# 4. a whole run, no steps, ROUTES_CHECK=0: PASS, RESTORED_OK, no routes.py, no receipts, test stop by hook x4
setup; run
[ "$(cat "$T/rc")" = 0 ] && grep -qx 'TP8 PASS; RESTORED_OK' "$T/runs/$RID/RESULT" 2>/dev/null && ok "run: TP8 PASS; RESTORED_OK" \
  || bad "run: TP8 PASS; RESTORED_OK [rc $(cat "$T/rc")]"
grep -q 'routes.py' "$M/calls" && bad "ROUTES_CHECK=0: routes.py not run" || ok "ROUTES_CHECK=0: routes.py not run"
[ ! -e "$T/runs/$RID/routes-receipts" ] && grep -q 'P0 OK: routes sha off' "$T/runs/$RID/window.log" && ok "ROUTES_CHECK=0: no receipts, sha off" \
  || bad "ROUTES_CHECK=0: no receipts, sha off"
[ "$(grep -c '^tverify' "$M/calls")" = 4 ] && [ "$(grep -c '^tstop' "$M/calls")" = 4 ] && ok "test verify x4 (P0), test stop x4 (restore)" \
  || bad "test verify x4 (P0), test stop x4 (restore)"
a=$(grep -n -m1 '^systemctl stop tp8-up-' "$M/calls" | cut -d: -f1); b=$(grep -n -m1 '^tstop' "$M/calls" | cut -d: -f1); [ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ] \
  && ok "restore: test launch unit stopped before TEST_STOP_CMD" || bad "restore: test launch unit stopped before TEST_STOP_CMD"
[ "$(grep -c '^tgr8 b. glm53 up' "$M/calls")" = 4 ] && [ "$(grep -c '^aup' "$M/calls")" = 4 ] && ok "test up x4, serve restored x4" \
  || bad "test up x4, serve restored x4"

# 5. steps pass: P2b runs on STEPS_HOST after P2 and before P3; bin check after it
setup; steps; run
[ "$(cat "$T/rc")" = 0 ] && ok "steps pass: run PASS" || bad "steps pass: run PASS [rc $(cat "$T/rc")]"
L=$T/runs/$RID/window.log
p2=$(grep -n 'P2 verify' "$L" | cut -d: -f1); p2b=$(grep -n 'P2b steps on b0' "$L" | cut -d: -f1); p3=$(grep -n 'P3 memclean' "$L" | cut -d: -f1)
[ -n "$p2b" ] && [ "$p2" -lt "$p2b" ] && [ "$p2b" -lt "$p3" ] && ok "P2b between P2 and P3" || bad "P2b between P2 and P3 [$p2 $p2b $p3]"
[ "$(grep -c '^steps b0' "$M/calls")" = 1 ] && ok "steps once, on STEPS_HOST" || bad "steps once, on STEPS_HOST"
s=$(grep -n -m1 '^steps b0' "$M/calls" | cut -d: -f1); m=$(grep -n -m1 '^md5bin' "$M/calls" | cut -d: -f1)
[ -n "$m" ] && [ "$s" -lt "$m" ] && ok "with steps: bin check after the steps, not in P0" || bad "with steps: bin check after the steps, not in P0 [$s $m]"

# 6. steps fail: no test up, run fails at P2b, serve restored
setup; steps false; run
[ "$(cat "$T/rc")" = 1 ] && grep -q 'finish: P2b: steps' "$T/runs/$RID/window.log" && ok "steps fail: finish P2b" \
  || bad "steps fail: finish P2b [rc $(cat "$T/rc")]"
! grep -q '^tgr8' "$M/calls" && grep -qx 'P2b: steps (rc 1); RESTORED_OK' "$T/runs/$RID/RESULT" && ok "steps fail: no test up, RESTORED_OK" \
  || bad "steps fail: no test up, RESTORED_OK"

# 7. steps pass but the boxes' bin/ differ: fails at P2b, no test up
setup; steps "touch $T/mock/bindiff"; run
grep -q 'finish: P2b: test bin/ differs' "$T/runs/$RID/window.log" && ! grep -q '^tgr8' "$M/calls" \
  && ok "bin/ differs after steps: no test up" || bad "bin/ differs after steps: no test up"

# 9. precheck pr95 B3: the restore completes during P2b (steps rc 0) -> the primary stops at the fence after P2b: no P3
# memclean/prep, no bin check, no test up on the restored boxes
rsc() { echo "bash $T/rec/\$TP8_RUN_ID/tp8-window.sh restore $T/rec/\$TP8_RUN_ID"; }   # the restore, run by STEPS_CMD
setup; steps "$(rsc)"; run
L=$T/runs/$RID/window.log
grep -q 'finish: fenced after P2b' "$L" && [ -e "$T/rec/$RID/RESTORED_OK" ] && ok "B3: restore during P2b -> fenced after P2b, RESTORED_OK" \
  || bad "B3: restore during P2b -> fenced after P2b [rc $(cat "$T/rc")]"
u=$(grep -n -m1 '^aup' "$M/calls" | cut -d: -f1)   # the restore's own memclean runs before its serve up
! grep -q 'P3 memclean' "$L" && [ -n "$u" ] && ! tail -n +"$u" "$M/calls" | grep -q '^memclean' && ! grep -q '^md5bin' "$M/calls" && ! grep -q '^tgr8' "$M/calls" \
  && ok "B3: no P3 memclean/prep, no bin check, no test up after the restore" || bad "B3: no P3 memclean/prep, no bin check, no test up after the restore"

# 10. precheck pr95 B2: the P2b units still run when the restore starts -> the restore stops the steps controller unit on
# STEPS_HOST, then tp8-steps-<run>-* on every box, then the test cluster, and only then brings the serve up
setup; steps "touch $T/mock/steps.live; $(rsc)"; run
c=$(grep -n -m1 "^systemctl stop tp8-steps-$RID.service" "$M/calls" | cut -d: -f1)
p=$(grep -n -m1 "^systemctl stop tp8-steps-$RID-\\*.service" "$M/calls" | cut -d: -f1)
t=$(grep -n -m1 '^tstop' "$M/calls" | cut -d: -f1); u=$(grep -n -m1 '^aup' "$M/calls" | cut -d: -f1)
[ -n "$c" ] && [ -n "$p" ] && [ -n "$t" ] && [ -n "$u" ] && [ "$c" -lt "$p" ] && [ "$p" -lt "$t" ] && [ "$t" -lt "$u" ] \
  && grep -q 'finish: fenced after P2b' "$T/runs/$RID/window.log" && ok "B2: steps ctl stop < per-box steps stop < test stop < serve up (restore inside P2b)" || bad "B2: order [$c $p $t $u]"
[ "$(grep -c "^systemctl stop tp8-steps-$RID.service" "$M/calls")" = 1 ] && [ "$(grep -c "^systemctl stop tp8-steps-$RID-\\*.service" "$M/calls")" = 4 ] \
  && [ -e "$T/rec/$RID/RESTORED_OK" ] && ok "B2: ctl stop once (STEPS_HOST), per-box stop x4, RESTORED_OK" || bad "B2: stop counts / RESTORED_OK"
# a steps unit that does not stop: no serve up, ALERT
setup; steps "touch $T/mock/steps.live $T/mock/steps.stuck; $(rsc)"; run
! grep -q '^aup' "$M/calls" && [ -e "$T/rec/$RID/ALERT" ] && [ ! -e "$T/rec/$RID/RESTORED_OK" ] && grep -q 'NOT STOPPED' "$T/rec/$RID/restore.log" 2>/dev/null \
  && ok "B2: stuck steps unit -> no serve up, ALERT" || bad "B2: stuck steps unit -> no serve up, ALERT"

# 11. precheck pr95v2 B3: the restore completes during the bin/ check after P2b -> fenced right before P3: no P3
setup; steps; echo "d=\$(ls -d $T/rec/*/ | head -1); bash \$d/tp8-window.sh restore \$d" > "$M/md5restore"; run
L=$T/runs/$RID/window.log; u=$(grep -n -m1 '^aup' "$M/calls" | cut -d: -f1)
grep -q 'finish: fenced before P3' "$L" && [ -e "$T/rec/$RID/RESTORED_OK" ] && [ -n "$u" ] && ! tail -n +"$u" "$M/calls" | grep -q '^memclean' \
  && ! grep -q '^tgr8' "$M/calls" && ok "B3v2: restore during the bin check -> fenced before P3, no memclean after the serve up" \
  || bad "B3v2: restore during the bin check -> fenced before P3 [rc $(cat "$T/rc")]"

# 12. precheck pr95v2 B2: the restore writes the per-host steps fence on every box BEFORE any steps unit stop
setup; steps "touch $T/mock/steps.live; $(rsc)"; run
f=$(grep -n -m1 -F "touch\\ $T/fence/tp8-steps-$RID.fence" "$M/calls" | cut -d: -f1); c=$(grep -n -m1 "^systemctl stop tp8-steps-$RID.service" "$M/calls" | cut -d: -f1)
[ -n "$f" ] && [ -n "$c" ] && [ "$f" -lt "$c" ] && [ "$(grep -F "touch\\ $T/fence/tp8-steps-$RID.fence" "$M/calls" | grep -c '^ssh b[0-3] ')" = 4 ] && [ -e "$T/fence/tp8-steps-$RID.fence" ] \
  && ok "B2v2: steps fence written on all 4 boxes before the steps unit stops (a unit registered earlier: caught by the scan)" \
  || bad "B2v2: steps fence before the unit stops [$f $c]"
# a steps unit registered after the fence: the controller unit refuses to start, STEPS_CMD never runs
setup; steps; bash "$W" cmds "$T/window.conf" > "$T/cmds" 2>&1; mkdir -p "$T/fence"; touch "$T/fence/tp8-steps-RUNID.fence"
c=$(awk -F'\t' '$1=="tp8-steps"{print $3}' "$T/cmds"); MOCK_HOST=b0 bash -c "$c" > /dev/null 2>&1
[ "$(/bin/cat /tmp/tp8-steps-RUNID.rc)" = 97 ] && ! grep -q '^steps ' "$M/calls" && grep -q 'steps fence .* is set' /tmp/tp8-steps-RUNID.log \
  && ok "B2v2: steps controller unit started after the fence: refuses (rc 97), STEPS_CMD not run" || bad "B2v2: fenced controller refuses"
rm -f /tmp/tp8-steps-RUNID.rc /tmp/tp8-steps-RUNID.log
# the fence write fails on a box: no serve up, ALERT
setup; steps "$(rsc)"; echo file > "$T/nofence"; sed -i "s#^STEPS_FENCE_DIR=.*#STEPS_FENCE_DIR=$T/nofence/x#" "$T/window.conf"; run
! grep -q '^aup' "$M/calls" && [ -e "$T/rec/$RID/ALERT" ] && [ ! -e "$T/rec/$RID/RESTORED_OK" ] && ! grep -q "^systemctl stop tp8-steps-$RID" "$M/calls" \
  && ok "B2v2: steps fence write fails -> no unit stop, no serve up, ALERT" || bad "B2v2: fence write fails -> ALERT"

# 8. defaults unchanged: no TEST_* / ROUTES_CHECK -> the tp8-ring launch.py stop and the routes dry run
setup; sed -i '/^TEST_STOP_CMD=/d;/^TEST_VERIFY_CMD=/d;/^ROUTES_CHECK=/d' "$T/window.conf"
bash "$W" cmds "$T/window.conf" > "$T/cmds" 2>&1
grep -qP '^tp8-stop\t.*launch.py down --local' "$T/cmds" && grep -qP '^tp8-p0-verify\t.*launch.py verify-down' "$T/cmds" \
  && ok "defaults: launch.py stop + verify-down" || bad "defaults: launch.py stop + verify-down"

# 9. HOLD=1 (standing serve): PASS keeps TP8 up, no restore; timer not verified stopped -> normal restore; restore RD later
setup; echo HOLD=1 >> "$T/window.conf"; run
[ "$(cat "$T/rc")" = 0 ] && grep -qx "TP8 PASS; HOLDING ($T/rec/$RID)" "$T/runs/$RID/RESULT" && [ -e "$T/rec/$RID/HOLDING" ] \
  && ok "HOLD: TP8 PASS; HOLDING, RD/HOLDING written" || bad "HOLD: TP8 PASS; HOLDING [rc $(cat "$T/rc") $(cat "$T/runs/$RID/RESULT" 2>/dev/null)]"
! grep -q 'restore-now' "$M/calls" && ! grep -q '^tstop' "$M/calls" && [ "$(grep -c '^aup' "$M/calls")" = 0 ] && [ -e "$M/tup" ] \
  && ok "HOLD: no restore, TP8 not stopped, serve not started" || bad "HOLD: no restore, TP8 not stopped, serve not started"
grep -q "systemctl stop tp8-restore-$RID.timer" "$M/calls" && grep -q "tp8-window.sh restore $T/rec/$RID" "$T/runs/$RID/window.log" \
  && ok "HOLD: timer stopped, fallback command printed" || bad "HOLD: timer stopped, fallback command printed"
: > "$M/calls"; bash "$W" restore "$T/rec/$RID" > "$T/out2" 2>&1
[ -e "$T/rec/$RID/RESTORED_OK" ] && [ "$(grep -c '^tstop' "$M/calls")" = 4 ] && [ "$(grep -c '^aup' "$M/calls")" = 4 ] && [ ! -e "$M/tup" ] \
  && ok "HOLD fallback: restore RD stops TP8 x4, serve up x4, RESTORED_OK" || bad "HOLD fallback: restore RD"
setup; echo HOLD=1 >> "$T/window.conf"; touch "$M/timer.stuck"; run
[ "$(cat "$T/rc")" = 0 ] && grep -qx 'TP8 PASS; RESTORED_OK' "$T/runs/$RID/RESULT" && [ ! -e "$T/rec/$RID/HOLDING" ] && grep -q "hold: timer 'active'" "$T/runs/$RID/window.log" \
  && ok "HOLD, timer not verified stopped: normal restore, RESTORED_OK" || bad "HOLD, timer stuck [rc $(cat "$T/rc") $(cat "$T/runs/$RID/RESULT" 2>/dev/null)]"
setup; echo HOLD=1 >> "$T/window.conf"; steps false; run
[ "$(cat "$T/rc")" = 1 ] && grep -qx 'P2b: steps (rc 1); RESTORED_OK' "$T/runs/$RID/RESULT" && [ ! -e "$T/rec/$RID/HOLDING" ] \
  && ok "HOLD, a failed run: restore as without HOLD" || bad "HOLD, failed run [rc $(cat "$T/rc")]"

echo "passed $pass, failed $failn"
[ $failn = 0 ]
