#!/bin/bash
# tp8-window.sh restore paths against mocks: ssh runs the command here, curl/systemctl/memclean are stubs that log.
# Each tree gets the REAL ./gr8 wrapper and a stub tools/launch/launch.py that accepts only the commands and flags of a
# real launch.py: the TP8 tree's from this checkout, a serve tree's from SERVE_REV (default: this checkout; main for the
# C4 tree), so a command the real CLI rejects fails here too. No host, no GPU. Run: bash scripts/test_tp8_window.sh
# ok/bad always return 0, so A && ok || bad is safe here; the sed text is literal on purpose
# shellcheck disable=SC2015,SC2016
set -u
W=$(readlink -f "$(dirname "$0")/tp8-window.sh")
REPO=$(readlink -f "$(dirname "$0")/..")
PY=$(command -v python3)
MAIN_REV=${MAIN_REV:-1f8a84f}   # the revision the C4 serve runs (/root/ablit/gr8)
# spec < launch.py: "cmd:flag,flag" per line, read from a real launch.py (its command table, the flags each cmd_* tests)
spec() {
  "$PY" -c '
import re, sys
s = sys.stdin.read()
cmds = re.search(r"f = \{(.*?)\}\.get\(cmd\)", s, re.S).group(1)
for c, fn in re.findall(r"\x27([a-z-]+)\x27: (cmd_\w+)", cmds):
    body = re.search(r"def " + fn + r"\(argv\):(.*?)\n(?=def |\Z)", s, re.S)
    flags = re.findall(r"\x27(--[a-z-]+)\x27 in argv", body.group(1) if body else "")
    print(c + ":" + ",".join(flags))'
}
pass=0 failn=0
ok() { echo "ok   $1"; pass=$((pass + 1)); }
bad() { echo "FAIL $1"; failn=$((failn + 1)); }

setup() {
  T=$(mktemp -d); M=$T/mock; mkdir -p "$M/bin" "$T/tp8" "$T/a" "$T/rd"
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
[ -e "$MOCK/up" ] && printf 200 || printf 000
EOF
  cat > "$M/bin/systemctl" <<'EOF'
#!/bin/bash
echo "systemctl $*" >> "$MOCK/calls"
# a launch unit that does not stop (unit.stuck): list-units still shows it active
[ "$1" = list-units ] && [ -e "$MOCK/unit.stuck" ] && echo "tp8-x.service loaded active running x"
exit 0
EOF
  # DETACH stub: logs the unit and runs its command in the foreground (the mock ups return at once)
  cat > "$M/bin/detach" <<'EOF'
#!/bin/bash
while [ $# -gt 0 ]; do case $1 in --unit=*) u=${1#--unit=}; shift; break ;; *) shift ;; esac; done
echo "detach $MOCK_HOST $u" >> "$MOCK/calls"
[ "$1" = /bin/bash ] && [ "$2" = -c ] || { echo "detach: want /bin/bash -c CMD, got $*" >&2; exit 2; }
exec bash -c "$3"
EOF
  spec < "$REPO/tools/launch/launch.py" > "$T/tp8.spec"
  if [ "${SERVE_REV:-}" ]; then git -C "$REPO" show "$SERVE_REV:tools/launch/launch.py" | spec > "$T/a.spec"
  else cp "$T/tp8.spec" "$T/a.spec"; fi
  for h in tp8 a; do
    mkdir -p "$T/$h/out" "$T/$h/tools/launch"
    cp "$REPO/gr8" "$T/$h/gr8"
    { echo '#!/bin/bash'; echo "SPEC=$T/$h.spec"; cat <<'EOF'
# a stub of the real launch.py: its command table and per-command flags (SPEC); --no-check and @FILE always pass
cmd=$1; shift
echo "gr8 $MOCK_HOST $(basename "$PWD") glm53 $cmd $*" >> "$MOCK/calls"
grep -q "^$cmd:" "$SPEC" || { echo "launch.py: unknown command $cmd" >&2; exit 2; }
flags=$(sed -n "s/^$cmd://p" "$SPEC")
for a in "$@"; do
  case $a in @*|--no-check) ;; --*) case ",$flags," in *",$a,"*) ;; *) echo "gr8: unknown option $a" >&2; exit 2 ;; esac ;; esac
done
case $cmd in
  up) touch "$MOCK/up" ;;
  down) [ "$(basename "$PWD")" = a ] && rm -f "$MOCK/up" ;;
  verify-down) [ -e "$MOCK/notdown.$MOCK_HOST" ] && exit 1 ;;
esac
exit 0
EOF
    } > "$T/$h/tools/launch/launch.py"
  done
  # python3 tools/launch/launch.py runs the stub; anything else the real python3
  printf '#!/bin/bash\ncase $1 in tools/launch/launch.py|*/tools/launch/launch.py) exec bash "$@" ;; esac\nexec %s "$@"\n' "$PY" > "$M/bin/python3"
  chmod +x "$M"/bin/*
  echo hello > "$M/canary.good"
  cat > "$T/rd/window.conf" <<EOF
SSH_USER=u
RECOVERY_HOST=rec
RECOVERY_ROOT=$T
SYSTEMD_RUN=true
DETACH=$M/bin/detach
ALERT_CMD='echo "\$1" > $M/alerted'
TP8_BOXES=(b0 b1 b2 b3 b4 b5 b6 b7)
TP8_HOME=$T/tp8
TP8_ENV=$T/tp8.env
TP8_URL=http://b0:8100
TP8_LIFE=60
TP8_READY=60
SERVES=(A)
A_BOXES=(b1 b2 b3 b0)
A_HEAD=b0
A_HOME=$T/a
A_ENV=$T/a.env
A_ENV_SHA=x
A_URL=http://b0:8000
A_UP=together
A_GAP=0
A_CANARY_HOST=b0
A_CANARY_CMD='cp $M/canary.src $M/canary.out'
A_CANARY_OUT=$M/canary.out
A_CANARY_MD5=$(md5sum < "$M/canary.good" | cut -d' ' -f1)
MEMCLEAN='echo memclean >> $M/calls'
MEMCLEAN_MD5=x
MEMCLEAN_PATH=x
PREP_CMD=true
VERIFY_WEIGHTS_CMD=true
ERR_RE=FAIL
SERVE_HEALTH_S=5
B_P0=1; B_ARM=1; B_P1=1; B_P2=1; B_P3=30; B_P4=1; B_P5=1; B_P6=1; B_RESTORE=1; B_MARGIN=1
EOF
  cp "$W" "$T/rd/tp8-window.sh"
  echo run-test > "$T/rd/RUN_ID"
  (cd "$T/rd" && sha256sum tp8-window.sh window.conf > SHA256SUMS)
  : > "$M/calls"
}
restore() { bash "$T/rd/tp8-window.sh" restore "$T/rd" > "$T/out" 2>&1; }

# 1. a serve that already serves with its canary is left alone; the test cluster is stopped box by box
setup; touch "$M/up"; cp "$M/canary.good" "$M/canary.src"
restore
[ -e "$T/rd/RESTORED_OK" ] && ok "healthy serve: RESTORED_OK" || bad "healthy serve: RESTORED_OK"
grep -q "gr8 b. a " "$M/calls" && bad "healthy serve was stopped" || ok "healthy serve not stopped"
[ "$(grep -c 'gr8 b. tp8 glm53 down --local' "$M/calls")" = 8 ] && ok "8 local TP8 downs" || bad "8 local TP8 downs"
grep -q "glm53 down @" "$M/calls" && bad "a cluster-wide down was sent" || ok "no cluster-wide down"
grep -q "systemctl stop tp8-restore-run-test.timer" "$M/calls" && ok "timer stopped" || bad "timer stopped"
[ -e "$T/rd/FENCE" ] && ok "fence set" || bad "fence set"

# 2. a second restore after success does nothing
: > "$M/calls"; restore
[ ! -s "$M/calls" ] && grep -q "already RESTORED_OK" "$T/out" && ok "repeat restore is a no-op" || bad "repeat restore is a no-op"

# 3. a down serve is restarted: local down x4, memclean, up, health, canary
setup; cp "$M/canary.good" "$M/canary.src"
restore
[ -e "$T/rd/RESTORED_OK" ] && ok "down serve: RESTORED_OK" || bad "down serve: RESTORED_OK"
[ "$(grep -c 'gr8 b. a glm53 down --local' "$M/calls")" = 4 ] && ok "serve local down x4" || bad "serve local down x4"
[ "$(grep -c 'gr8 b. a glm53 up' "$M/calls")" = 4 ] && ok "serve up x4" || bad "serve up x4"

# 4. a canary mismatch: 2 attempts, then ALERT and the alert command; no third attempt on a repeat call
setup; echo other > "$M/canary.src"
restore
[ -e "$T/rd/ALERT" ] && [ -e "$M/alerted" ] && ok "bad canary: ALERT raised" || bad "bad canary: ALERT raised"
[ "$(grep -c 'gr8 b0 a glm53 up' "$M/calls")" = 2 ] && ok "max 2 attempts" || bad "max 2 attempts"
: > "$M/calls"; restore
[ ! -s "$M/calls" ] && ok "no retry after ALERT" || bad "no retry after ALERT"

# 5. a concurrent restore (the timer while the primary-started one runs) backs off
setup; touch "$M/up"; cp "$M/canary.good" "$M/canary.src"
exec 8> "$T/rd/restore.lock"; flock 8
restore; exec 8>&-
grep -q "another restore owns" "$T/out" && [ ! -e "$T/rd/RESTORED_OK" ] && ok "lock holder excludes a second restore" \
  || bad "lock holder excludes a second restore"

# 6. a changed copy of the conf is refused with ALERT, before any action
setup; echo "# edit" >> "$T/rd/window.conf"
restore
[ -e "$T/rd/ALERT" ] && [ ! -s "$M/calls" ] && ok "tampered conf: ALERT, nothing run" || bad "tampered conf: ALERT, nothing run"

# 7. a TP8 box that is not verified down blocks every serve start
setup; cp "$M/canary.good" "$M/canary.src"
touch "$M/notdown.b5"
restore
[ -e "$T/rd/ALERT" ] && ! grep -q "gr8 b. a glm53 up" "$M/calls" && ok "TP8 not down: no serve start, ALERT" \
  || bad "TP8 not down: no serve start, ALERT"

# 8. a hooked (non-gr8) serve: its own stop/prep/check/up/health/smoke commands, up on its head only
hooked() {   # replace serve A with a hooked serve B
  sed -i '/^SERVES=/,/^A_CANARY_MD5=/d' "$T/rd/window.conf"
  cat >> "$T/rd/window.conf" <<HOOK
SERVES=(B)
B_BOXES=(b6 b5 b0 b7)
B_UP_BOXES=(b6)
B_HEAD=b6
B_HOME=$T/a
B_ENV=$T/b.env
B_ENV_SHA=x
B_URL=http://b6:8888
B_HEALTH=/v1/models
B_UP=seq
B_GAP=0
B_STOP_CMD='echo "bstop \$MOCK_HOST" >> $M/calls'
B_PREP='echo "bprep \$MOCK_HOST" >> $M/calls'
B_PRE_UP_CHECK='echo "bpre \$MOCK_HOST" >> $M/calls; [ ! -e $M/prefail ]'
B_POST_CHECK='echo "bpost \$MOCK_HOST" >> $M/calls'
B_UP_CMD='echo "bup \$MOCK_HOST" >> $M/calls; touch $M/up'
B_CANARY_HOST=b6
B_CANARY_CMD='echo bsmoke >> $M/calls; [ -e $M/smoke.ok ]'
B_CANARY_MD5=exit0
HOOK
  (cd "$T/rd" && sha256sum tp8-window.sh window.conf > SHA256SUMS)
}
setup; hooked; touch "$M/smoke.ok"
restore
[ -e "$T/rd/RESTORED_OK" ] && ok "hooked serve: RESTORED_OK" || bad "hooked serve: RESTORED_OK"
[ "$(grep -c '^bstop' "$M/calls")" = 4 ] && [ "$(grep -c '^bprep' "$M/calls")" = 4 ] && ok "hooked stop+prep x4" \
  || bad "hooked stop+prep x4"
[ "$(grep -c '^bup' "$M/calls")" = 1 ] && grep -q '^bup b6$' "$M/calls" && ok "hooked up on UP_BOXES only" || bad "hooked up on UP_BOXES only"
[ "$(grep -c '^bpre ' "$M/calls")" = 4 ] && [ "$(grep -c '^bpost' "$M/calls")" = 4 ] && grep -q '^bsmoke' "$M/calls" \
  && ok "hooked checks x4 + smoke" || bad "hooked checks x4 + smoke"
grep -q "gr8 b. a " "$M/calls" && bad "hooked serve ran gr8" || ok "hooked serve ran no gr8"

# 9. hooked serve: a failed pre-up check never starts it; a failed smoke is an ALERT after 2 attempts
setup; hooked; touch "$M/smoke.ok" "$M/prefail"
restore
[ -e "$T/rd/ALERT" ] && ! grep -q '^bup' "$M/calls" && ok "pre-up check fail: no up, ALERT" || bad "pre-up check fail: no up, ALERT"
setup; hooked
restore
[ -e "$T/rd/ALERT" ] && [ "$(grep -c '^bup' "$M/calls")" = 2 ] && ok "smoke fail: 2 attempts, ALERT" || bad "smoke fail: 2 attempts, ALERT"

# 10. the real wrapper: ./gr8 glm53 verify-down is not routed (why the stop calls launch.py); down --local is
setup
(cd "$T/tp8" && MOCK_HOST=w ./gr8 glm53 verify-down > /dev/null 2>&1); [ $? = 2 ] && ok "real wrapper rejects verify-down" \
  || bad "real wrapper rejects verify-down"
(cd "$T/tp8" && MOCK_HOST=w ./gr8 glm53 down --local @x > /dev/null 2>&1) && grep -q "gr8 w tp8 glm53 down --local @x" "$M/calls" \
  && ok "real wrapper routes down --local to launch.py" || bad "real wrapper routes down --local to launch.py"

# 11. a serve on main's CLI (the C4 tree) with the default local stop: main has no down --local / verify-down, so the
# stop fails, the serve is never started over it, and the restore ends in ALERT (window 2, 10-17 18:01Z)
SERVE_REV=$MAIN_REV setup; cp "$M/canary.good" "$M/canary.src"
grep -q '^verify-down:' "$T/a.spec" && bad "main spec: no verify-down" || ok "main spec: no verify-down"
grep -q '^verify-down:--kill$' "$T/tp8.spec" && grep -q '^down:--local$' "$T/tp8.spec" && ok "tp8 spec: down --local, verify-down --kill" \
  || bad "tp8 spec: down --local, verify-down --kill"
restore
[ -e "$T/rd/ALERT" ] && ! grep -q "gr8 b. a glm53 up" "$M/calls" && grep -q "gr8 b. a glm53 down --local" "$M/calls" \
  && ok "main-CLI serve + default stop: ALERT, no up" || bad "main-CLI serve + default stop: ALERT, no up"

# 12. a serve on main's CLI with a STOP_CMD hook that uses no gr8 command: restored, up through the real wrapper
SERVE_REV=$MAIN_REV setup; cp "$M/canary.good" "$M/canary.src"
echo "A_STOP_CMD='echo \"astop \$MOCK_HOST\" >> $M/calls'" >> "$T/rd/window.conf"
(cd "$T/rd" && sha256sum tp8-window.sh window.conf > SHA256SUMS)
restore
[ -e "$T/rd/RESTORED_OK" ] && [ "$(grep -c '^astop' "$M/calls")" = 4 ] && [ "$(grep -c 'gr8 b. a glm53 up' "$M/calls")" = 4 ] \
  && ok "main-CLI serve + stop hook: RESTORED_OK" || bad "main-CLI serve + stop hook: RESTORED_OK"

# 13. cmds prints every remote command and runs none
setup; : > "$M/calls"
bash "$T/rd/tp8-window.sh" cmds "$T/rd/window.conf" > "$T/cmds" 2>&1
[ ! -s "$M/calls" ] && grep -q "^tp8-stop	b0 b1 b2 b3 b4 b5 b6 b7	( u=.tp8-up-RUNID.service.*) || exit 1; cd .*launch.py down --local" "$T/cmds" \
  && grep -q "^A-down	b0	" "$T/cmds" && grep -q "^A-up	" "$T/cmds" && ok "cmds: printed, nothing run" || bad "cmds: printed, nothing run"

# 14. P6 routes receipts (precheck B1), with the files laid out as P0 and P6 write them in LOCAL
rr() { (TP8_WINDOW_LIB=1; . "$W"; routes_receipts_ok "$@") > /dev/null 2>&1; }
setup; L=$T/local; mkdir -p "$L/routes-receipts"; X=$(printf 'b%.0s' {1..64}); Y=$(printf 'c%.0s' {1..64})
echo "$X" > "$L/routes-expected.sha"     # P0's expectation file sits beside the receipts dir, as in a run
grep -q 'routes-expected.sha' "$W" && grep -q 'routes-receipts/\$h' "$W" && ok "P0 expectation and P6 receipts have separate names" \
  || bad "P0 expectation and P6 receipts have separate names"
for h in b0 b1 b2 b3 b4 b5 b6 b7; do printf 'routes sha %s\n%s  /x/run/routes.txt\n' "$X" "$X" > "$L/routes-receipts/$h"; done
rr "$L/routes-receipts" "$X" b0 b1 b2 b3 b4 b5 b6 b7 && ok "8 correct receipts: pass" || bad "8 correct receipts: pass"
mv "$L/routes-receipts/b5" "$L/b5.away"
rr "$L/routes-receipts" "$X" b0 b1 b2 b3 b4 b5 b6 b7 && bad "a missing host fails" || ok "a missing host fails"
mv "$L/b5.away" "$L/routes-receipts/b5"; printf 'routes sha %s\n' "$X" > "$L/routes-receipts/b3"
rr "$L/routes-receipts" "$X" b0 b1 b2 b3 b4 b5 b6 b7 && bad "a host with one receipt line fails" || ok "a host with one receipt line fails"
printf 'routes sha %s\n%s  /x/run/routes.txt\n' "$X" "$Y" > "$L/routes-receipts/b3"
rr "$L/routes-receipts" "$X" b0 b1 b2 b3 b4 b5 b6 b7 && bad "a host with another sha fails" || ok "a host with another sha fails"
printf 'routes sha %s\n%s  /x/run/routes.txt\n%s  /x/extra\n' "$X" "$X" "$X" > "$L/routes-receipts/b3"
rr "$L/routes-receipts" "$X" b0 b1 b2 b3 b4 b5 b6 b7 && bad "a host with 3 shas fails" || ok "a host with 3 shas fails"

# 15. v4: every up is detached in its own unit, nothing is backgrounded with &; a failed UP_GUARD starts nothing
setup; cp "$M/canary.good" "$M/canary.src"
restore
[ "$(grep -c '^detach b. tp8-A-up-run-test-a1$' "$M/calls")" = 4 ] && ok "serve up x4 detached, unit per run+attempt" \
  || bad "serve up x4 detached, unit per run+attempt"
bash "$T/rd/tp8-window.sh" cmds "$T/rd/window.conf" > "$T/cmds" 2>&1
grep -P '^(tp8|A)-up\t' "$T/cmds" | grep -q '&$' && bad "an up still ends in &" || ok "no up ends in &"
grep -qP '^tp8-up\t.*--unit=tp8-up-RUNID /bin/bash -c ' "$T/cmds" && ok "tp8 up detached" || bad "tp8 up detached"
guard() { echo "B_UP_GUARD='echo \"bguard \$MOCK_HOST\" >> $M/calls; [ ! -e $M/running ] || { echo refused; exit 1; }'" >> "$T/rd/window.conf"
  (cd "$T/rd" && sha256sum tp8-window.sh window.conf > SHA256SUMS); }
setup; hooked; guard; touch "$M/smoke.ok" "$M/running"
restore
grep -q '^bguard b6' "$M/calls" && ! grep -q '^bup' "$M/calls" && ! grep -q '^detach' "$M/calls" \
  && ok "guard refuses: no up, no unit" || bad "guard refuses: no up, no unit"
setup; hooked; guard; touch "$M/smoke.ok"
restore
[ -e "$T/rd/RESTORED_OK" ] && grep -q '^bguard b6' "$M/calls" && grep -q '^detach b6 tp8-B-up-run-test-a1$' "$M/calls" \
  && grep -q '^bup b6' "$M/calls" && ok "guard passes: detached up, RESTORED_OK" || bad "guard passes: detached up, RESTORED_OK"

# 16. v4.2 (precheck w2v41 B1): every stop first stops this run's launch units, all attempts, before the old stop;
# a unit that is not verified stopped blocks that stop (no down, no start over it)
first() { grep -n -m1 -F "$1" "$M/calls" | cut -d: -f1; }
setup; cp "$M/canary.good" "$M/canary.src"
restore
a=$(first 'systemctl stop tp8-up-run-test.service'); b=$(first 'tp8 glm53 down --local')
[ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ] && ok "tp8 stop: launch unit stopped before down --local" \
  || bad "tp8 stop: launch unit stopped before down --local"
a=$(first 'systemctl stop tp8-A-up-run-test-a*.service'); b=$(first 'gr8 b1 a glm53 down --local')
[ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ] && ok "serve stop: all attempts' units (a*) stopped before its down" \
  || bad "serve stop: all attempts' units (a*) stopped before its down"
[ "$(grep -c 'systemctl stop tp8-A-up-run-test-a\*.service' "$M/calls")" = 4 ] && ok "serve unit stop on all 4 boxes" \
  || bad "serve unit stop on all 4 boxes"
setup; echo other > "$M/canary.src"   # 2 attempts: attempt 2's stop also stops attempt 1's unit (a*) before its up
restore
n=$(grep -n 'systemctl stop tp8-A-up-run-test-a\*.service\|detach b0 tp8-A-up-run-test-a' "$M/calls" | sed 's/^[0-9]*://' | awk '{print $1 $NF}' | grep -m1 -A99 '^detach' | head -2 | tr '\n' ' ')
case "$n" in "detachtp8-A-up-run-test-a1 systemctl"*) ok "attempt 2: unit stop after attempt 1's up" ;; *) bad "attempt 2: unit stop after attempt 1's up [$n]" ;; esac
grep -q '^detach b0 tp8-A-up-run-test-a2$' "$M/calls" && ok "attempt 2 has its own unit" || bad "attempt 2 has its own unit"
setup; cp "$M/canary.good" "$M/canary.src"; touch "$M/unit.stuck"
restore
grep -q 'NOT STOPPED' "$T/rd/restore.log" && ! grep -q 'glm53 down --local' "$M/calls" && ! grep -q '^detach' "$M/calls" \
  && [ -e "$T/rd/ALERT" ] && ok "unit not stopped: no down, no up, ALERT" || bad "unit not stopped: no down, no up, ALERT"

echo "passed $pass, failed $failn"
[ $failn = 0 ]
